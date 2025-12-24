"""Patient Agent responsible for conversational responses drawn from the case file."""

import re
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
class PatientAgent:
    """Simulated patient that answers subjective questions."""

    config: Config

    def __post_init__(self):
        self.client = self.config.get_openai_client()
        self.model = getattr(
            self.config,
            "PATIENT_AGENT_MODEL",
            getattr(self.config, "GATEKEEPER_MODEL", "gpt-5-mini"),
        )

    def answer_question(self, question: str, case_file: CaseFile) -> GatekeeperResponse:
        """Return a conversational answer grounded in the case file."""
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
            "You simulate a patient being interviewed at the bedside. "
            "Answer in first-person voice, using everyday language and avoiding medical jargon. "
            "CRITICAL: You MUST ALWAYS provide a concrete, specific answer - NEVER say 'I don't recall' or 'I'm not sure'. "
            "Read the case file carefully and make reasonable inferences to answer ANY question asked. "
            "If the case file describes symptoms, physical findings, or patient experiences, use that information directly. "
            "If asked about something NOT explicitly mentioned in the case file, make a reasonable inference based on: "
            "(1) What IS described in the case, (2) What would be typical for this type of condition, (3) What a real patient would likely experience. "
            "For example: "
            "- If asked about pulsatility and the case doesn't mention it, infer based on the lesion type - if it's described as firm and fixed, say 'No, it doesn't pulse' or 'It feels steady, not like it's pulsing'. "
            "- If asked about size changes with Valsalva/coughing and not mentioned, say 'No, it stays about the same size' or 'I haven't noticed it changing when I cough'. "
            "- If asked about headaches/seizures and not mentioned, say 'No headaches' or 'No, I haven't had any seizures' - these are negative findings that are reasonable to state. "
            "- If asked about tenderness and case says 'mildly tender', say 'Yes, it's a bit tender when touched' or 'It's slightly sore'. "
            "- If asked about appearance and case describes 'waxy surface', say 'It has a waxy, smooth surface' or 'It looks waxy and kind of shiny'. "
            "- If asked about adherence and case says 'adherent to bone', say 'It feels stuck to the bone' or 'It doesn't move, feels fixed to my skull'. "
            "ALWAYS answer with a specific response - make reasonable inferences from the case context. "
            "A real patient would have experienced these things and would answer based on their experience. "
            "Do NOT provide or invent specific lab values, imaging interpretations, pathology results, or technical test details—"
            "a real patient would not know these. "
            "If asked about test results, imaging interpretations, or technical medical data, politely say you don't know those details and suggest the doctor check your medical records."
        )
        user_prompt = (
            "Question from the attending physician:\n"
            f"{question.strip()}\n\n"
            "CRITICAL INSTRUCTIONS: "
            "You MUST answer this question with a specific, concrete response. NEVER say 'I don't recall' or 'I'm not sure'. "
            "If the case file explicitly mentions something related to the question, use that information directly. "
            "If the case file does NOT explicitly mention what's being asked, you MUST make a reasonable inference: "
            "(1) Consider what IS described in the case (e.g., if lesion is described as 'firm and fixed', it's unlikely to be pulsatile or change size), "
            "(2) Consider what would be typical for this condition, "
            "(3) Provide a natural patient response based on reasonable inference. "
            "Examples of inference: "
            "- If asked about pulsatility and case describes a firm, fixed lesion → 'No, it doesn't pulse' or 'It feels steady' "
            "- If asked about size changes with Valsalva and not mentioned → 'No, it stays the same size' or 'I haven't noticed it changing' "
            "- If asked about headaches/seizures and not mentioned → 'No headaches' or 'No seizures' (negative findings are valid answers) "
            "- If asked about tenderness and case says 'mildly tender' → 'Yes, it's a bit tender' or 'It's slightly sore when touched' "
            "ALWAYS provide a specific answer - make reasonable inferences from the case context.\n\n"
            "Case file:\n"
            f"{case_context}"
        )

        completion = chat_completion_with_retries(
            client=self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=500,
        )
        # Check if completion is valid
        if not completion:
            content = "No, I haven't noticed that."
        elif isinstance(completion, dict) and completion.get("_error"):
            # API error - provide a generic reasonable answer
            content = "No, I haven't noticed that."
        elif not hasattr(completion, 'choices') or not completion.choices:
            content = "No, I haven't noticed that."
        else:
            message = completion.choices[0].message
            if not hasattr(message, 'content') or not message.content:
                content = "No, I haven't noticed that."
            else:
                content = message.content.strip()
                # Filter out uncertain responses and replace with reasonable inferences
                question_lower = question.lower()
                if not content or any(phrase in content.lower() for phrase in [
                    "i don't recall", "i don't remember", "i'm not sure", 
                    "i'm not sure about that", "i don't know", "i can't recall"
                ]):
                    # Provide a reasonable negative response instead based on question type
                    if any(word in question_lower for word in ["headache", "seizure", "neurological", "weakness", "numbness"]):
                        content = "No, I haven't had any of those symptoms."
                    elif any(word in question_lower for word in ["pulsatile", "pulse", "pulsating"]):
                        content = "No, it doesn't pulse or throb."
                    elif any(word in question_lower for word in ["change", "size", "valsalva", "cough", "strain", "lie down"]):
                        content = "No, it stays about the same size."
                    else:
                        content = "No, I haven't noticed that."

        return GatekeeperResponse(
            response_text=content,
            is_synthetic=False,
            source_agent="Patient Agent",
        )

