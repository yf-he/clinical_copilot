"""Configuration settings for SDBench."""

import os
from typing import Optional
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for SDBench."""

    # Provider selection: 'openai' or 'openrouter'
    API_PROVIDER: str = os.getenv("SDBENCH_API_PROVIDER", "openrouter").lower()
    print(API_PROVIDER)
    # OpenAI API settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # OpenRouter API settings
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # Model settings (OpenRouter model IDs by default)
    # You can override with env: SDBENCH_*_MODEL variables
    # Fix Gatekeeper/Judge on 4o-mini unless explicitly overridden in code
    GATEKEEPER_MODEL: str = os.getenv("SDBENCH_GATEKEEPER_MODEL", "openai/gpt-4o-mini")
    PATIENT_AGENT_MODEL: str = os.getenv("SDBENCH_PATIENT_MODEL", GATEKEEPER_MODEL)
    EXAMINATION_AGENT_MODEL: str = os.getenv("SDBENCH_EXAM_MODEL", GATEKEEPER_MODEL)
    JUDGE_MODEL: str = os.getenv("SDBENCH_JUDGE_MODEL", "openai/gpt-4o-mini")

    # Azure OpenAI API settings
    AZURE_OPENAI_API_KEY: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_BASE_URL: str = os.getenv("AZURE_OPENAI_BASE_URL", "https://medevalkit.openai.azure.com/")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    # Cost settings
    PHYSICIAN_VISIT_COST: float = 300.0

    # Evaluation settings
    CORRECT_DIAGNOSIS_THRESHOLD: int = 4  # Score >= 4 is considered correct

    # Data settings
    VALIDATION_CASES: int = 248
    TEST_CASES: int = 56

    @classmethod
    def get_openai_client(cls) -> OpenAI:
        """Return an OpenAI-compatible client for the configured provider."""
        if cls.API_PROVIDER == "openrouter":
            if not cls.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter")
            return OpenAI(base_url=cls.OPENROUTER_BASE_URL, api_key=cls.OPENROUTER_API_KEY)
        elif cls.API_PROVIDER == "azureopenai":
            if not cls.AZURE_OPENAI_API_KEY:
                raise ValueError("AZURE_OPENAI_API_KEY environment variable is required for Azure OpenAI")
            return AzureOpenAI(
                    api_version=cls.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=cls.AZURE_OPENAI_BASE_URL,
                    api_key=cls.AZURE_OPENAI_API_KEY,
                )

        # default to OpenAI
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        return OpenAI(api_key=cls.OPENAI_API_KEY)

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present for the chosen provider."""
        if cls.API_PROVIDER == "openrouter":
            if not cls.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY environment variable is required")
        else:
            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable is required")
        return True
