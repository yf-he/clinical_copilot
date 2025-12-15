"""Patient Agent responsible for conversational responses drawn from the case file."""

from dataclasses import dataclass

from config import Config
from data_models import CaseFile, GatekeeperResponse
from utils.llm_client import chat_completion_with_retries

MAX_CONTEXT_CHARS = 8000


def _trim_case_text(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Trim very long case files to keep prompts bounded."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


@dataclass
class PatientAgent:
    """Simulated patient that answers subjective questions."""

    config: Config

    def __post_init__(self):
        self.client = self.config.get_openai_client()
        self.model = getattr(
            self.config,
            "PATIENT_AGENT_MODEL",
            getattr(self.config, "GATEKEEPER_MODEL", "openai/gpt-4o-mini"),
        )

    def answer_question(self, question: str, case_file: CaseFile) -> GatekeeperResponse:
        """Return a conversational answer grounded in the case file."""
        case_context = f"""Patient Case File
--------------------
Initial Abstract:
{case_file.initial_abstract}

Full Case Narrative:
{_trim_case_text(case_file.full_case_text)}
"""

        system_prompt = (
            "You simulate a patient being interviewed at the bedside. "
            "Answer in first-person voice, using everyday language and avoiding medical jargon. "
            "Only use details that are explicitly present in the provided patient file. "
            "Do NOT provide or invent lab values, imaging reads, pathology, or any test/consult results—"
            "a real patient would not know them. "
            "If asked for test results, imaging interpretations, or any chart-only data, politely say you "
            "do not know and direct the clinician to the medical record or care team. "
            "If the file lacks the requested subjective detail, state politely that you were not told."
        )
        user_prompt = (
            "Question from the attending physician:\n"
            f"{question.strip()}\n\n"
            "Reference case file (only source of truth):\n"
            f"{case_context}"
        )

        completion = chat_completion_with_retries(
            client=self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=320,
        )
        content = completion.choices[0].message.content.strip()
        if not content:
            content = "I'm sorry, I wasn't given that information."

        return GatekeeperResponse(
            response_text=content,
            is_synthetic=False,
            source_agent="Patient Agent",
        )

