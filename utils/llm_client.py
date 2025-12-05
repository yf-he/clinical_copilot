import time
import os
import traceback
from typing import Mapping, List, Dict, Any, Optional

from openai import OpenAI, AzureOpenAI

from config import Config


def get_client_from_config(config: Optional[Config] = None) -> OpenAI:
    cfg = config or Config()
    return cfg.get_openai_client()


def chat_completion_with_retries(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_retries: int = 5,
    retry_interval_sec: int = 20,
    **kwargs: Any,
) -> Mapping:
    # Convert max_tokens to max_completion_tokens for Azure OpenAI
    is_azure_openai = isinstance(client, AzureOpenAI)
    if is_azure_openai and 'max_tokens' in kwargs:
        kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
    if model.startswith('gpt-5'):
        if is_azure_openai and 'temperature' in kwargs:
            kwargs.pop('temperature')
    
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
        except Exception as e:  # Broad catch to handle provider SDK differences
            last_err = e
            remaining = max_retries - attempt - 1
            if remaining <= 0:
                break
            # Verbose diagnostics
            print("LLM request failed:", flush=True)
            print(f"  ErrorType: {type(e).__name__}", flush=True)
            print(f"  ErrorRepr: {e!r}", flush=True)
            print(f"  Error: {e}", flush=True)
            print(f'  Model: {model}', flush=True)
            # Some SDK errors may have status/response
            status = getattr(e, 'status_code', None) or getattr(e, 'status', None)
            if status is not None:
                print(f"  HTTPStatus: {status}", flush=True)
            resp = getattr(e, 'response', None)
            if resp is not None:
                try:
                    print(f"  Response: {resp}", flush=True)
                except Exception:
                    pass
            if os.getenv('SDBENCH_DEBUG', '0') in ('1','true','True','YES','yes'):
                traceback.print_exc()
            print(
                f"Retry in {retry_interval_sec}s... ({remaining} retries left)",
                flush=True,
            )
            time.sleep(retry_interval_sec)
    if last_err:
        print("LLM request ultimately failed:", flush=True)
        print(f"  ErrorType: {type(last_err).__name__}", flush=True)
        print(f"  ErrorRepr: {last_err!r}", flush=True)
        status = getattr(last_err, 'status_code', None) or getattr(last_err, 'status', None)
        if status is not None:
            print(f"  HTTPStatus: {status}", flush=True)
        resp = getattr(last_err, 'response', None)
        if resp is not None:
            try:
                print(f"  Response: {resp}", flush=True)
            except Exception:
                pass
        if os.getenv('SDBENCH_DEBUG', '0') in ('1','true','True','YES','yes'):
            traceback.print_exc()
    return {}


def truncate_text(encoding, text: str, max_tokens: int) -> str:
    if not text:
        return text
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        print(f"WARNING: Maximum token length exceeded ({len(tokens)} > {max_tokens})")
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text


