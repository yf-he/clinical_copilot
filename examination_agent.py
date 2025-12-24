"""Examination Agent that surfaces objective findings/tests from the case file."""

import re
from dataclasses import dataclass

from config import Config
from data_models import CaseFile, GatekeeperResponse
from utils.llm_client import chat_completion_with_retries

MAX_CONTEXT_CHARS = 8000


def _trim_case_text(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def _sanitize_case_text(case_text: str) -> str:
    """Remove diagnosis options and ground truth from case text to prevent agents from seeing answers."""
    # Remove the OPTIONS section if present
    lines = case_text.split('\n')
    sanitized_lines = []
    
    for line in lines:
        # Stop at OPTIONS section
        if line.strip().upper() == "OPTIONS":
            break
        # Skip lines that look like option labels (A., B., C., D.)
        if re.match(r'^[A-D][\.\):]\s+', line.strip(), re.IGNORECASE):
            continue
        sanitized_lines.append(line)
    
    return '\n'.join(sanitized_lines).strip()


@dataclass
class ExaminationAgent:
    """Simulated clinical information system for objective data."""

    config: Config

    def __post_init__(self):
        self.client = self.config.get_openai_client()
        self.model = getattr(
            self.config,
            "EXAMINATION_AGENT_MODEL",
            getattr(self.config, "GATEKEEPER_MODEL", "gpt-5-mini"),
        )

    def fetch_result(self, request: str, case_file: CaseFile) -> GatekeeperResponse:
        """Return objective data for ordered tests or examinations."""
        # Sanitize case text to remove diagnosis options and ground truth
        sanitized_case_text = _sanitize_case_text(case_file.full_case_text)
        case_context = f"""Patient Case File
--------------------
Initial Abstract:
{case_file.initial_abstract}

Full Case Narrative:
{_trim_case_text(sanitized_case_text)}
"""

        system_prompt = (
            "You act as a clinical information system that surfaces only objective data. "
            "Return concise, factual findings pulled directly from the patient file. "
            "Format results the way a lab report or imaging summary would appear. "
            "If the requested data is missing, explicitly state that it is not documented."
        )
        user_prompt = (
            "Ordered test or exam request:\n"
            f"{request.strip()}\n\n"
            "Patient case file (authoritative source):\n"
            f"{case_context}"
        )

        completion = chat_completion_with_retries(
            client=self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        content = completion.choices[0].message.content.strip()
        if not content:
            content = "Requested study is not documented in the chart."

        return GatekeeperResponse(
            response_text=content,
            is_synthetic=False,
            source_agent="Examination Agent",
        )

