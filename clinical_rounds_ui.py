"""
Multi-turn Medical Diagnosis Copilot

This interactive demo simulates a real-world clinical diagnostic scenario where:
- You play the role of an attending physician making diagnostic decisions
- An AI resident doctor (LLM) can assist by drafting questions, test orders, or diagnoses
- The Medical Evidence System provides patient information, test results, and clinical data
- Each step builds toward a final diagnosis that is evaluated against ground truth

The scenario mimics a hospital bedside encounter where physicians iteratively:
1. Ask questions about patient history and symptoms
2. Order diagnostic tests and imaging studies
3. Synthesize information to reach a diagnosis

You can choose to draft each step yourself or let the AI resident doctor suggest the next action,
creating a collaborative human-AI diagnostic workflow.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

import streamlit as st

from config import Config
from data_loader import CaseFile, load_jsonl_cases
from patient_agent import PatientAgent
from examination_agent import ExaminationAgent
from judge_agent import JudgeAgent
from cost_estimator import CostEstimator
from data_models import ActionType, AgentAction, GatekeeperResponse
from utils.llm_client import chat_completion_with_retries

DEFAULT_DATASET = Path(
    "C:/Users/t-yufeihe/Downloads/SDBench-main/SDBench-main/converted/converted/test-00000-of-00001.jsonl"
)


@st.cache_resource
def load_cases(dataset_path: str) -> List[CaseFile]:
    dataset_path = str(Path(dataset_path).expanduser())
    return load_jsonl_cases(dataset_path, publication_year=2025, is_test_case=True)


def initialize_state():
    if "case" not in st.session_state:
        st.session_state.case = None
    if "encounter" not in st.session_state:
        st.session_state.encounter = None
    if "actions" not in st.session_state:
        st.session_state.actions = []
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "action_authors" not in st.session_state:
        st.session_state.action_authors = []
    if "judge_score" not in st.session_state:
        st.session_state.judge_score = None
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "action_content_input" not in st.session_state:
        st.session_state.action_content_input = ""
    if "action_content_pending" not in st.session_state:
        st.session_state.action_content_pending = None
    if "llm_status" not in st.session_state:
        st.session_state.llm_status = ""
    if "action_type_label" not in st.session_state:
        st.session_state.action_type_label = "ask question"
    if "action_type_label_pending" not in st.session_state:
        st.session_state.action_type_label_pending = None


def normalize_diagnosis_text(content: str, case: CaseFile) -> str:
    text = (content or "").strip()
    if not text:
        return text
    options = getattr(case, "diagnosis_options", []) or []
    for idx, option in enumerate(options):
        label = chr(65 + idx)
        for sep in (".", ")", ":"):
            prefix = f"{label}{sep}"
            if text.lower().startswith(prefix.lower()):
                remainder = text[len(prefix):].lstrip()
                if remainder.lower() == option.lower():
                    return option
                return remainder or option
    if re.match(r"^[A-Da-d][\.\):]\s*", text):
        text = re.sub(r"^[A-Da-d][\.\):]\s*", "", text, count=1).strip()
    return text


def add_action(
    action: AgentAction,
    patient_agent: PatientAgent,
    examination_agent: ExaminationAgent,
    case: CaseFile,
    cost_estimator: CostEstimator,
    author_label: str,
) -> None:
    st.session_state.actions.append(action)
    st.session_state.action_authors.append(author_label)

    if action.action_type == ActionType.DIAGNOSE:
        st.session_state.responses.append(
            GatekeeperResponse(
                response_text="Diagnosis submitted for evaluation.",
                source_agent="Attending Physician",
            )
        )
        return

    if action.action_type == ActionType.ASK_QUESTIONS:
        response = patient_agent.answer_question(action.content, case)
    elif action.action_type == ActionType.REQUEST_TESTS:
        response = examination_agent.fetch_result(action.content, case)
    else:
        raise ValueError(f"Unsupported action type {action.action_type}")

    st.session_state.responses.append(response)

    if action.action_type == ActionType.REQUEST_TESTS:
        test_cost = cost_estimator.calculate_test_cost(action.content)
        st.session_state.total_cost += test_cost
        st.info(f"Estimated test cost: ${test_cost:.2f}")


def finalize_diagnosis(judge: JudgeAgent, case: CaseFile):
    final_action = next(
        (a for a in reversed(st.session_state.actions) if a.action_type == ActionType.DIAGNOSE),
        None,
    )
    if not final_action:
        st.warning("Submit a diagnosis before finalizing rounds.")
        return
    st.session_state.judge_score = judge.evaluate_diagnosis(final_action.content, case)


def build_clinical_context(case: CaseFile) -> str:
    lines = [
        f"Patient ID: {case.case_id}",
        f"Chief Concern Summary: {case.initial_abstract}",
        "",
        "Running Encounter:",
    ]
    for idx, action in enumerate(st.session_state.actions, 1):
        author = st.session_state.action_authors[idx - 1] if idx - 1 < len(st.session_state.action_authors) else "Attending"
        lines.append(f"{idx}. {author} [{action.action_type.value}]: {action.content}")
        if idx - 1 < len(st.session_state.responses):
            resp = st.session_state.responses[idx - 1]
            responder = resp.source_agent or "Medical Evidence System"
            lines.append(f"   {responder}: {resp.response_text}")
    return "\n".join(lines)


def suggest_action_with_llm(
    desired_action: ActionType,
    case: CaseFile,
    cfg: Config,
    model_name: str,
    temperature: float = 0.3,
) -> Optional[str]:
    if not case:
        return None

    client = cfg.get_openai_client()
    context = build_clinical_context(case)
    diagnosis_options = getattr(case, "diagnosis_options", [])

    task_map = {
        ActionType.ASK_QUESTIONS: (
            "Draft a single, clinically precise bedside question for the patient or staff "
            "that would meaningfully advance the diagnostic work-up. "
            "CRITICAL: Review all previous questions in the encounter history - do NOT ask a similar or related question. "
            "If a question was already asked (even if unanswered), choose a completely different question or consider switching to requesting tests. "
            "IMPORTANT: Ask about SUBJECTIVE symptoms, sensations, history, or patient experience. "
            "Do NOT ask about test results, imaging studies, lab values, or whether prior tests/imaging are available - these should be ordered as tests, not asked as questions."
        ),
        ActionType.REQUEST_TESTS: (
            "Order exactly one high-yield diagnostic test or imaging study. Include modality and any pertinent qualifiers. "
            "CRITICAL: Review all previous test orders in the encounter history - do NOT order a similar test. "
            "For example, if 'MRI brain' was already ordered, do NOT order 'MRI brain with contrast' or 'MRI brain and scalp' - these are considered the same test. "
            "Choose a completely different test that has NOT been ordered yet."
        ),
        ActionType.DIAGNOSE: (
            "Provide a concise, definitive diagnosis (or leading impression) that best explains the presentation."
        ),
    }

    system_prompt = (
        "You are an AI resident doctor assisting an attending physician in clinical diagnosis. "
        "CRITICAL: Review all previous actions in the encounter history before drafting. "
        "NEVER repeat similar tests or questions - if a similar action was already taken, choose something completely different. "
        "Keep outputs to a single actionable sentence."
    )

    user_prompt = f"""
{context}

Task: {task_map[desired_action]}
Respond with only the requested utterance, no preamble.
"""
    if desired_action == ActionType.DIAGNOSE and diagnosis_options:
        option_lines = "\n".join(
            f"{chr(65 + idx)}. {opt}" for idx, opt in enumerate(diagnosis_options)
        )
        user_prompt += f"""

Diagnosis options (choose the single best answer and quote it verbatim; reply with the text only, no option letters):
{option_lines}
"""

    try:
        completion = chat_completion_with_retries(
            client=client,
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=temperature,
            max_tokens=4000,
        )
        
        # Check if completion is valid (chat_completion_with_retries returns {} or error dict on failure)
        # On success, OpenAI returns a response object with .choices attribute
        # On failure, it returns an empty dict {} or dict with _error key
        if not completion:
            raise ValueError("Empty response from AI model - check API connection and model availability")
        
        # Check if it's a failed response (dict with _error or empty dict) vs successful response object
        if isinstance(completion, dict):
            if completion.get("_error"):
                # This is the failure case with error info
                error_msg = completion.get("_error", "Unknown error")
                error_type = completion.get("_error_type", "Unknown")
                raise ValueError(f"API request failed: {error_type} - {error_msg}. Check API connection, model availability ({model_name}), and API keys.")
            elif not completion.get("choices"):
                # Empty dict returned by chat_completion_with_retries
                raise ValueError("API request failed after retries - check API connection, model availability, and API keys")
        
        # Check if completion has choices attribute (successful response object)
        if not hasattr(completion, 'choices'):
            raise ValueError("Invalid response structure from AI model")
        
        if not completion.choices or len(completion.choices) == 0:
            raise ValueError("No choices in AI model response - model may have failed to generate output")
        
        message = completion.choices[0].message
        if not hasattr(message, 'content') or message.content is None:
            # Check if there's finish_reason that might explain the issue
            finish_reason = getattr(completion.choices[0], 'finish_reason', None)
            if finish_reason == 'length':
                raise ValueError("Response was truncated - try increasing max_tokens (current: 400)")
            elif finish_reason:
                raise ValueError(f"Model stopped generating: finish_reason={finish_reason}")
            raise ValueError("Empty content in AI model response - model may have failed to generate output")
        
        content = message.content.strip()
        if not content:
            # Check finish_reason for debugging
            finish_reason = getattr(completion.choices[0], 'finish_reason', None)
            usage_info = getattr(completion, 'usage', None)
            debug_info = f"finish_reason={finish_reason}"
            if usage_info:
                debug_info += f", tokens_used={getattr(usage_info, 'total_tokens', 'N/A')}"
            raise ValueError(f"Empty response content from AI model ({debug_info}) - try increasing max_tokens or check if model supports the request format")
        
        return content
    except Exception as exc:
        st.error(f"AI resident could not draft a response: {exc}")
        return None


def decide_action_with_llm(
    case: CaseFile,
    cfg: Config,
    model_name: str,
    temperature: float = 0.2,
) -> Optional[tuple[ActionType, str]]:
    """Let the AI resident choose the action type and draft the content."""
    if not case:
        return None

    client = cfg.get_openai_client()
    context = build_clinical_context(case)
    diagnosis_options = getattr(case, "diagnosis_options", [])

    user_prompt = f"""
{context}

Select the single best next clinical action (ask_questions, request_tests, or diagnose).

ACTION SELECTION PRIORITY (in order):
1. **PREFER ask_questions** - Ask clinical questions to gather more information before ordering tests. Only order tests if you have already asked several relevant questions or if a test is clearly the next critical step.
2. **request_tests** - Only order tests when you have sufficient information from questions, or when a specific test is urgently needed for diagnosis.
3. **diagnose** - Only diagnose when you have gathered enough information through questions and/or tests.

CRITICAL CONSTRAINTS - READ ALL PREVIOUS ACTIONS FIRST:
- **NEVER repeat or request similar tests/questions already in the encounter history above.**
- **PREFER asking questions** - Before ordering expensive tests, ask relevant clinical questions first to narrow down the differential.
- **For questions**: Ask about SUBJECTIVE symptoms, sensations, patient experience, history, or physical characteristics the patient can describe. 
  **DO NOT ask about test results, imaging studies, lab values, or whether prior tests/imaging are available** - these should be ordered as tests, not asked as questions.
- **For test orders**: If a similar test was already ordered (e.g., "MRI brain" vs "MRI brain with contrast"), choose a DIFFERENT test or action type entirely. Consider asking a question instead.
- **For questions**: If a similar question was already asked, choose a completely different question or switch to requesting tests/diagnosis.
- **Review the entire encounter history** - if your proposed action is similar to any previous action, choose something different.
- Keep the content one concise sentence.
- If ordering a test, include modality/qualifiers; choose the single highest-yield test that has NOT been ordered yet.
- If diagnosing, provide the leading diagnosis (or pick the best option). Reply with text only—no option letters.

Return a JSON object with keys "action_type" and "content". "action_type" must be one of ["ask_questions","request_tests","diagnose"]. Respond with JSON only.
"""
    if diagnosis_options:
        option_lines = "\n".join(f"{chr(65 + idx)}. {opt}" for idx, opt in enumerate(diagnosis_options))
        user_prompt += f"""

Diagnosis options:
{option_lines}
"""

    try:
        completion = chat_completion_with_retries(
            client=client,
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI resident doctor assisting the attending. "
                        "CRITICAL: Before choosing an action, carefully review ALL previous actions in the encounter history. "
                        "PREFER asking clinical questions over ordering tests - gather information through questions first before requesting expensive tests. "
                        "NEVER repeat similar tests or questions - if 'MRI brain' was ordered, do NOT order 'MRI brain with contrast' or similar variants. "
                        "If a similar action was already taken, choose something completely different. "
                        "Keep the output minimal and respond with ONLY valid JSON, no markdown formatting or extra text."
                    ),
                },
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=temperature,
            max_tokens=5000,
        )
        
        # Check if completion is valid (chat_completion_with_retries returns {} or error dict on failure)
        # On success, OpenAI returns a response object with .choices attribute
        # On failure, it returns an empty dict {} or dict with _error key
        if not completion:
            raise ValueError("Empty response from AI model - check API connection and model availability")
        
        # Check if it's a failed response (dict with _error or empty dict) vs successful response object
        if isinstance(completion, dict):
            if completion.get("_error"):
                # This is the failure case with error info
                error_msg = completion.get("_error", "Unknown error")
                error_type = completion.get("_error_type", "Unknown")
                raise ValueError(f"API request failed: {error_type} - {error_msg}. Check API connection, model availability ({model_name}), and API keys.")
            elif not completion.get("choices"):
                # Empty dict returned by chat_completion_with_retries
                raise ValueError("API request failed after retries - check API connection, model availability, and API keys")
        
        # Check if completion has choices attribute (successful response object)
        if not hasattr(completion, 'choices'):
            raise ValueError("Invalid response structure from AI model")
        
        if not completion.choices or len(completion.choices) == 0:
            raise ValueError("No choices in AI model response - model may have failed to generate output")
        
        message = completion.choices[0].message
        if not hasattr(message, 'content') or message.content is None:
            # Check if there's finish_reason that might explain the issue
            finish_reason = getattr(completion.choices[0], 'finish_reason', None)
            if finish_reason == 'length':
                raise ValueError("Response was truncated - try increasing max_tokens (current: 500)")
            elif finish_reason:
                raise ValueError(f"Model stopped generating: finish_reason={finish_reason}")
            raise ValueError("Empty content in AI model response - model may have failed to generate output")
        
        raw = message.content.strip()
        if not raw:
            # Check finish_reason for debugging
            finish_reason = getattr(completion.choices[0], 'finish_reason', None)
            usage_info = getattr(completion, 'usage', None)
            debug_info = f"finish_reason={finish_reason}"
            if usage_info:
                debug_info += f", tokens_used={getattr(usage_info, 'total_tokens', 'N/A')}"
            raise ValueError(f"Empty response content from AI model ({debug_info}) - try increasing max_tokens or check if model supports the request format")
        
        # Try to extract JSON from markdown code blocks if present
        json_text = raw
        if "```json" in raw:
            # Extract JSON from ```json ... ``` block
            start = raw.find("```json") + 7
            end = raw.find("```", start)
            if end != -1:
                json_text = raw[start:end].strip()
        elif "```" in raw:
            # Extract JSON from ``` ... ``` block
            start = raw.find("```") + 3
            end = raw.find("```", start)
            if end != -1:
                json_text = raw[start:end].strip()
        else:
            # Try to find JSON object in the text
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_text = raw[start:end+1]
        
        payload = json.loads(json_text)
        action_value = payload.get("action_type")
        content = (payload.get("content") or "").strip()
        if action_value not in {a.value for a in ActionType} or not content:
            raise ValueError("Missing or invalid fields in AI response.")
        action_type = ActionType(action_value)
        return action_type, content
    except json.JSONDecodeError as exc:
        st.error(f"AI resident returned invalid JSON. Raw response: {raw[:200] if 'raw' in locals() else 'N/A'}")
        return None
    except Exception as exc:
        st.error(f"AI resident could not select an action: {exc}")
        return None


def build_transcript(case: CaseFile) -> str:
    lines = []
    judge_score = st.session_state.judge_score

    lines.append("========== Multi-turn Medical Diagnosis Copilot - Clinical Encounter Transcript ==========\n")
    lines.append(f"Patient ID: {case.case_id}")
    lines.append("Care Team: Attending Physician + AI Resident Doctor")
    lines.append(f"Chief Concern: {case.initial_abstract}\n")

    for idx, action in enumerate(st.session_state.actions, 1):
        author = st.session_state.action_authors[idx - 1] if idx - 1 < len(st.session_state.action_authors) else "Attending"
        lines.append(f"---------- ROUND {idx} ----------")
        lines.append(f"[{author}] ({action.action_type.value})")
        lines.append(action.content)
        if idx <= len(st.session_state.responses):
            resp = st.session_state.responses[idx - 1]
            responder = resp.source_agent or "Medical Evidence System"
            lines.append(f"[{responder}]")
            lines.append(resp.response_text)

    lines.append("\n----------------------------------------")
    lines.append(f"[Resource Stewardship] Total estimated cost: ${st.session_state.total_cost:.2f}")

    if judge_score:
        lines.append("========================================")
        lines.append("[ATTENDING BOARD REVIEW]")
        lines.append(f"Score: {judge_score.score}/5")
        lines.append(f"Label: {judge_score.label}")
        lines.append("Reasoning:")
        lines.append(judge_score.reasoning)

    lines.append("========================================")
    lines.append("[REFERENCE DIAGNOSIS]")
    lines.append(case.ground_truth_diagnosis)
    lines.append("----------------------------------------")
    lines.append("[FULL CASE DATA]")
    lines.append(case.full_case_text)
    return "\n".join(lines)


def main():
    st.set_page_config(
        page_title="Clinical Copilots in Action: A Multi-Turn Diagnosis Demo for Human–AI Collaboration",
        layout="wide",
    )
    st.title("Clinical Copilots in Action: A Multi-Turn Diagnosis Demo for Human–AI Collaboration")

    st.markdown(
        """
### Scenario
You play the attending physician running bedside rounds on a simulated patient. On every turn you may ask the patient direct questions, order specific diagnostic tests, or submit a working diagnosis. Each action is logged, priced, and answered by grounded agents that reveal information only when clinically appropriate.

### Task Flow
- **Step 1 – Load a Case:** Use the sidebar to pick a dataset and patient chart.
- **Step 2 – Review & Plan:** Read the *Patient Snapshot* and choose whether to ask a question, request a test, or submit a diagnosis. Decide if you or the AI Resident drafts the step.
- **Step 3 – Draft (Optional):** Click *Let AI Resident Doctor draft* to auto-populate the action field with an editable suggestion.
- **Step 4 – Submit Actions:** Send the action to the Medical Evidence System. Patient-focused queries route to the Patient Agent, while objective data requests invoke the Examination Agent and Cost Estimator.
- **Step 5 – Iterate & Finalize:** Inspect the *Clinical Timeline*, track cumulative cost, and when satisfied, press *Finalize diagnosis & request attending review* to receive the Judge Agent’s score and rationale.
"""
    )

    with st.expander("📋 About & Architecture", expanded=True):
        screenshot_path = Path("latex/figs/screenshot.jpg")
        if screenshot_path.exists():
            st.image(str(screenshot_path), caption="Clinical Copilot system demo")

        system_tab, agents_tab = st.tabs(["System Operation", "Agent Layer"])
        with system_tab:
            st.markdown(
                """
**Turn-based Workflow**

1. *Initialization:* Select a `CaseFile` and review the chief concern.
2. *Plan the next action:* Choose `ASK_QUESTIONS`, `REQUEST_TESTS`, or `DIAGNOSE`, and decide whether you or the AI Resident drafts it.
3. *AI draft (optional):* The AI Resident Doctor consumes the entire encounter transcript to propose a single actionable utterance that you can edit.
4. *Action submission:* The action routes to the correct backend agent.
5. *Backend processing:*  
   - Questions → Patient Agent  
   - Test orders → Examination Agent (plus Cost Estimator)  
   - Diagnoses → Stored for Judge review
6. *State refresh:* The UI appends the new turn and any returned evidence to the clinical timeline.
7. *Finalization:* Trigger the Judge Agent to compare your final diagnosis with ground truth and return a 1–5 score and rationale.
"""
            )
        with agents_tab:
            st.markdown(
                """
**Modular Components**

- **User Interface Layer:** Streamlit UI that manages state, renders the patient snapshot, and coordinates each turn.
- **Data Layer (`CaseFile`):** JSONL-backed records containing the full hidden case narrative, diagnosis options, and gold-label diagnosis.
- **Agent Layer:**  
  - *AI Resident Doctor:* Drafts questions, test orders, or diagnoses on demand. Can be swapped for any LLM/toolchain by changing the model ID.  
  - *Patient Agent:* Answers bedside questions in a first-person voice using only the ground-truth file.  
  - *Examination Agent:* Returns objective findings (labs, imaging, physical exam) in report style.  
  - *Judge Agent:* Scores the final diagnosis with an LLM rubric.  
  - *Cost Estimator:* Tracks cumulative resource use with CPT-style heuristics.
"""
            )

    st.divider()

    initialize_state()

    with st.sidebar:
        st.header("Clinical Setup")
        dataset_path = st.text_input("Dataset (.sdbench.jsonl)", value=str(DEFAULT_DATASET))
        ai_model_id = st.text_input("AI Resident Doctor Model", value="gpt-5-mini")
        if st.button("Reset Encounter"):
            for key in [
                "case",
                "encounter",
                "actions",
                "responses",
                "action_authors",
                "judge_score",
                "total_cost",
                "action_content_input",
                "action_content_pending",
                "llm_status",
                "action_type_label",
                "action_type_label_pending",
            ]:
                st.session_state.pop(key, None)
            st.rerun()

    if not dataset_path:
        st.stop()

    try:
        cases = load_cases(dataset_path)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    case_ids = [case.case_id for case in cases]
    selected_case = st.selectbox("Select patient chart", case_ids, index=0 if case_ids else None)
    if selected_case:
        st.session_state.case = next(c for c in cases if c.case_id == selected_case)

    cfg = Config()
    patient_agent = PatientAgent(cfg)
    examination_agent = ExaminationAgent(cfg)
    judge = JudgeAgent(cfg)
    cost_estimator = CostEstimator(cfg)

    case = st.session_state.case
    if not case:
        st.info("Select a case to begin rounds.")
        st.stop()

    st.subheader("Patient Snapshot")
    st.write(case.initial_abstract)

    if st.session_state.action_content_pending is not None:
        st.session_state.action_content_input = st.session_state.action_content_pending
        st.session_state.action_content_pending = None

    # Handle pending action type label before creating the widget
    action_options = ["ask question", "request test", "diagnose"]
    if st.session_state.action_type_label_pending is not None:
        pending_label = st.session_state.action_type_label_pending
        if pending_label in action_options:
            st.session_state.action_type_label = pending_label
        st.session_state.action_type_label_pending = None

    st.subheader("Plan the Next Step")
    action_col, submit_col = st.columns([4, 1])
    action_type_label = action_col.selectbox(
        "Clinical action",
        action_options,
        format_func=lambda x: x.title(),
        key="action_type_label",
    )
    actor_label = action_col.radio(
        "Who drafts this step?",
        ["Attending (You)", "AI Resident Doctor (LLM)"],
        horizontal=True,
    )

    action_type_map = {
        "ask question": ActionType.ASK_QUESTIONS,
        "request test": ActionType.REQUEST_TESTS,
        "diagnose": ActionType.DIAGNOSE,
    }
    desired_action = action_type_map[action_type_label]
    diagnosis_options = getattr(case, "diagnosis_options", [])

    if actor_label == "AI Resident Doctor (LLM)":
        if action_col.button("AI Resident: Choose Action & Draft", use_container_width=False):
            with st.spinner("AI Resident Doctor choosing action..."):
                decision = decide_action_with_llm(case, cfg, ai_model_id)
            if decision:
                decided_action, suggestion = decision
                target_label = {
                    ActionType.ASK_QUESTIONS: "ask question",
                    ActionType.REQUEST_TESTS: "request test",
                    ActionType.DIAGNOSE: "diagnose",
                }[decided_action]
                if decided_action == ActionType.DIAGNOSE:
                    suggestion = normalize_diagnosis_text(suggestion, case)
                st.session_state.action_type_label_pending = target_label
                st.session_state.action_content_pending = suggestion
                st.session_state.llm_status = (
                    f"Drafted and action selected by AI Resident Doctor ({ai_model_id}) → {target_label.title()}"
                )
                st.rerun()
        if action_col.button("AI Resident: Draft Content Only", use_container_width=False):
            with st.spinner("AI Resident Doctor drafting..."):
                suggestion = suggest_action_with_llm(desired_action, case, cfg, ai_model_id)
            if suggestion:
                if desired_action == ActionType.DIAGNOSE:
                    suggestion = normalize_diagnosis_text(suggestion, case)
                st.session_state.action_content_pending = suggestion
                st.session_state.llm_status = f"Drafted by AI Resident Doctor ({ai_model_id})"
                st.rerun()

    if desired_action == ActionType.DIAGNOSE:
        current_text = st.session_state.get("action_content_input", "")
        cleaned = normalize_diagnosis_text(current_text, case)
        if cleaned != current_text:
            st.session_state.action_content_input = cleaned

    content = action_col.text_area(
        "Document your bedside question/order/impression",
        key="action_content_input",
        height=180,
    )

    if st.session_state.llm_status:
        st.info(st.session_state.llm_status)

    if submit_col.button("Submit clinical action", use_container_width=True):
        final_content = st.session_state.action_content_input.strip()
        if not final_content:
            st.warning("Enter or generate content before submitting.")
        else:
            if desired_action == ActionType.DIAGNOSE:
                final_content = normalize_diagnosis_text(final_content, case)
            author = "AI Resident Doctor" if actor_label == "AI Resident Doctor (LLM)" else "Attending"
            action = AgentAction(action_type=desired_action, content=final_content)
            add_action(
                action,
                patient_agent,
                examination_agent,
                case,
                cost_estimator,
                author,
            )
            st.session_state.action_content_pending = ""
            st.session_state.llm_status = ""
            st.rerun()

    if st.session_state.actions:
        st.subheader("Clinical Timeline")
        for idx, action in enumerate(st.session_state.actions, 1):
            author = st.session_state.action_authors[idx - 1]
            with st.expander(f"Round {idx}: {author} • {action.action_type.value.replace('_', ' ').title()}"):
                st.markdown(f"**{author}:** {action.content}")
                if idx <= len(st.session_state.responses):
                    responder = (
                        st.session_state.responses[idx - 1].source_agent
                        or "Medical Evidence System"
                    )
                    st.markdown(
                        f"**{responder}:** {st.session_state.responses[idx - 1].response_text}"
                    )

    st.markdown(f"**Cumulative Estimated Cost:** ${st.session_state.total_cost:.2f}")

    if st.button("Finalize diagnosis & request attending review"):
        finalize_diagnosis(judge, case)
        st.rerun()

    if st.session_state.judge_score:
        st.subheader("Attending Board Review")
        st.markdown(
            f"""
- **Score:** {st.session_state.judge_score.score}/5
- **Label:** {st.session_state.judge_score.label}
- **Reasoning:** {st.session_state.judge_score.reasoning}
"""
        )
        st.subheader("Reference Diagnosis")
        st.markdown(f"**{case.ground_truth_diagnosis}**")
        st.subheader("Encounter Transcript")
        transcript_text = build_transcript(case)
        st.code(transcript_text, language="markdown")

    st.subheader("Download Complete Transcript")
    transcript_text = build_transcript(case)
    st.download_button(
        label="Download Clinical Transcript",
        data=transcript_text,
        file_name=f"{case.case_id}_MultiTurnDiagnosisCopilot.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()

