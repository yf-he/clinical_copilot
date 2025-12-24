"""
Streamlit UI to run the LLMDiagnosticAgent on a selected case and
visualize the multi-turn dialog with the MAI-DxO system.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from config import Config
from sdbench import SDBench
from data_loader import load_jsonl_cases
from example_agents import LLMDiagnosticAgent

DEFAULT_DATASET = "/Users/yufei/Desktop/SDBench/converted/test-00000-of-00001.jsonl"


@st.cache_resource
def load_cases(dataset_path: str):
    dataset_path = str(Path(dataset_path).expanduser())
    cases = load_jsonl_cases(dataset_path, publication_year=2025, is_test_case=True)
    return cases


def run_simulation(dataset_path: str, case_id: str, model_id: str):
    cfg = Config()
    bench = SDBench(cfg)

    cases = load_cases(dataset_path)
    case_lookup = {case.case_id: case for case in cases}
    if case_id not in case_lookup:
        raise ValueError(f"Case ID {case_id} not found in dataset.")
    case = case_lookup[case_id]

    agent = LLMDiagnosticAgent(name=f"LLM({model_id})", config=cfg)
    agent.model = model_id

    transcripts_dir = tempfile.mkdtemp(prefix="llm_ui_transcripts_")
    encounter = bench.run_single_encounter(
        agent,
        case,
        max_turns=20,
        disable_cost=False,
        transcript_dir=transcripts_dir,
    )

    sanitized_name = agent.name.replace("/", "_").replace(":", "_").replace(" ", "_").replace("(", "_").replace(")", "_")
    transcript_path = Path(transcripts_dir) / f"{case.case_id}_{sanitized_name}.txt"
    transcript_text = transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else "Transcript file not found."

    return encounter, transcript_text, transcripts_dir


def main():
    st.set_page_config(page_title="LLM Agent Diagnostic UI", layout="wide")
    st.title("LLM Agent Multi-Turn Diagnostic Explorer")

    # Session state
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "encounter" not in st.session_state:
        st.session_state.encounter = None
    if "transcript_text" not in st.session_state:
        st.session_state.transcript_text = ""
    if "transcript_dir" not in st.session_state:
        st.session_state.transcript_dir = ""
    if "turn_index" not in st.session_state:
        st.session_state.turn_index = 0
    if "turn_blocks" not in st.session_state:
        st.session_state.turn_blocks = []
    if "sub_index" not in st.session_state:
        st.session_state.sub_index = 0

    # Step 1: Select dataset/case/model
    with st.expander("Step 1 - Select dataset and case", expanded=(st.session_state.step == 1)):
        dataset_path = st.text_input("Dataset (.sdbench.jsonl)", value=DEFAULT_DATASET, key="dataset_path")
        cases = []
        if dataset_path:
            try:
                cases = load_cases(dataset_path)
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
        case_ids = [case.case_id for case in cases]
        selected_case = st.selectbox("Select Case ID", case_ids, index=0 if case_ids else None, key="selected_case")
        model_id = st.text_input("Agent Model ID", value="gpt-5-mini", key="model_id")
        col1, col2 = st.columns([1, 1])
        if col1.button("Next ➜", use_container_width=True) and selected_case:
            st.session_state.step = 2
        if col2.button("Reset", use_container_width=True):
            for k in ["step", "encounter", "transcript_text", "transcript_dir", "turn_index", "turn_blocks"]:
                st.session_state.pop(k, None)
            st.rerun()

    # Step 2: Run simulation (once)
    if st.session_state.step >= 2 and st.session_state.encounter is None:
        with st.expander("Step 2 - Run simulation", expanded=True):
            with st.spinner("Running simulation..."):
                try:
                    enc, t_text, t_dir = run_simulation(st.session_state.dataset_path, st.session_state.selected_case, st.session_state.model_id)
                    st.session_state.encounter = enc
                    st.session_state.transcript_text = t_text
                    st.session_state.transcript_dir = t_dir
                    # Parse transcript into blocks by turns
                    blocks = []
                    current = []
                    for line in t_text.splitlines():
                        if line.startswith("---------- TURN "):
                            if current:
                                blocks.append("\n".join(current).strip())
                                current = []
                        current.append(line)
                    if current:
                        blocks.append("\n".join(current).strip())
                    st.session_state.turn_blocks = blocks
                    st.session_state.turn_index = 0
                    st.success("Simulation completed.")
                    st.session_state.step = 3
                except Exception as exc:
                    st.error(f"Simulation failed: {exc}")

    # Step 3: Reveal turns incrementally
    if st.session_state.step >= 3 and st.session_state.encounter is not None:
        enc = st.session_state.encounter

        st.subheader("Case Summary")
        st.markdown(
            f"""
- **Case ID:** {enc.case_id}
- **Agent:** LLM({st.session_state.model_id})
- **Total Turns:** {len(enc.actions)}
- **Diagnosis:** {enc.final_diagnosis or "Not provided"}
- **Judge Score:** {enc.judge_score.score if enc.judge_score else "N/A"}
- **Judge Label:** {enc.judge_score.label if enc.judge_score else "N/A"}
- **Estimated Total Cost:** ${enc.total_cost:.2f}
"""
        )

        st.subheader("Transcript - Step by Step")
        total_blocks = len(st.session_state.turn_blocks)
        shown = st.session_state.turn_index

        col_prev, col_next, col_all = st.columns([1, 1, 1])
        if col_prev.button("◀ Previous", use_container_width=True) and shown > 0:
            st.session_state.turn_index -= 1
            st.session_state.sub_index = 0
            st.rerun()
        if col_next.button("Next ▶", use_container_width=True) and shown < total_blocks:
            st.session_state.turn_index += 1
            st.session_state.sub_index = 0
            st.rerun()
        if col_all.button("Show All", use_container_width=True):
            st.session_state.turn_index = total_blocks
            st.rerun()

        # If there are no detected turns, show the full transcript as a single block
        if total_blocks == 0 and st.session_state.transcript_text:
            st.code(st.session_state.transcript_text, language="markdown")
        else:
            # Render full previous turns entirely
            for i in range(min(max(shown - 1, 0), total_blocks)):
                label = "Initial Abstract" if i == 0 else f"Turn {i}"
                st.markdown(f"**{label}**")
                st.code(st.session_state.turn_blocks[i], language="markdown")

            # For the current (last) visible turn, reveal Agent/Gatekeeper pieces step-by-step
            if shown > 0:
                current_idx = shown - 1
                current_block = st.session_state.turn_blocks[current_idx]
                # Split sub-steps by major headers inside a turn
                sub_steps = []
                sub_current = []
                for line in current_block.splitlines():
                    if line.startswith("[Agent:") or line.startswith("[Gatekeeper") or line.startswith("[MAI-DxO"):
                        if sub_current:
                            sub_steps.append("\n".join(sub_current).strip())
                            sub_current = []
                    sub_current.append(line)
                if sub_current:
                    sub_steps.append("\n".join(sub_current).strip())

                total_sub = len(sub_steps)
                col_s_prev, col_s_next, col_s_all = st.columns([1, 1, 1])
                if col_s_prev.button("◀ Prev Step", use_container_width=True, key="prev_sub") and st.session_state.sub_index > 0:
                    st.session_state.sub_index -= 1
                    st.rerun()
                if col_s_next.button("Next Step ▶", use_container_width=True, key="next_sub") and st.session_state.sub_index < total_sub:
                    st.session_state.sub_index += 1
                    st.rerun()
                if col_s_all.button("Show All Steps", use_container_width=True, key="all_sub"):
                    st.session_state.sub_index = total_sub
                    st.rerun()

                current_label = "Initial Abstract" if current_idx == 0 else f"Turn {current_idx}"
                st.markdown(f"**{current_label} (Steps {min(st.session_state.sub_index, total_sub)} / {total_sub})**")
                for s in range(min(st.session_state.sub_index, total_sub)):
                    st.code(sub_steps[s], language="markdown")

        st.subheader("Download")
        st.download_button(
            label="Download Full Transcript",
            data=st.session_state.transcript_text,
            file_name=f"{enc.case_id}_LLM_{st.session_state.model_id.replace('/', '_')}.txt",
            mime="text/plain",
        )
        st.caption(f"Transcript directory: {st.session_state.transcript_dir}")


if __name__ == "__main__":
    main()


