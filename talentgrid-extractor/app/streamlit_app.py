"""
TalentGrid Structured Extraction Demo
Paste or upload resume/cover-letter text, run it through the baseline
(prompt-only) pipeline and the fine-tuned model, and compare the two
extractions side by side with confidence indicators.

Run:
    streamlit run app/streamlit_app.py

Notes:
- Loads both models lazily and caches them so switching between resumes in
  a session doesn't reload weights each time.
- If no fine-tuned adapter is found at ADAPTER_DIR, the app still runs in
  "baseline only" mode with a clear banner, so the app is demoable even
  before training finishes.
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from src.baseline import load_model, run_extraction, BASE_MODEL
from src.fine_tuned_inference import load_fine_tuned, run_extraction_ft

ADAPTER_DIR = "adapters/qwen2.5-1.5b-extraction-lora"

st.set_page_config(page_title="TalentGrid Extraction Demo", layout="wide")
st.title("TalentGrid — Structured Extraction: Baseline vs. Fine-Tuned")
st.caption(
    "Paste resume/cover-letter text below. Both models run on the same "
    f"base weights ({BASE_MODEL}) — the only difference is the LoRA "
    "fine-tune, so any gap you see below is attributable to fine-tuning, "
    "not a different base model."
)


@st.cache_resource(show_spinner="Loading baseline model...")
def get_baseline():
    return load_model(BASE_MODEL)


@st.cache_resource(show_spinner="Loading fine-tuned adapter...")
def get_fine_tuned():
    if not Path(ADAPTER_DIR).exists():
        return None, None
    return load_fine_tuned(ADAPTER_DIR)


def confidence_badge(conf: float) -> str:
    if conf is None:
        return "⚪ n/a"
    if conf >= 0.75:
        return f"🟢 {conf:.2f}"
    if conf >= 0.5:
        return f"🟡 {conf:.2f}"
    return f"🔴 {conf:.2f}"


def render_extraction(col, title, result: dict | None, raw: str = ""):
    with col:
        st.subheader(title)
        if result is None:
            st.error("Model did not return valid JSON for this input.")
            with st.expander("Raw output"):
                st.code(raw)
            return

        for field_name, label in [
            ("skills", "Skills"), ("education", "Education"),
            ("certifications", "Certifications"), ("career_gaps", "Career Gaps"),
        ]:
            items = result.get(field_name, [])
            st.markdown(f"**{label}** ({len(items)})")
            if not items:
                st.caption("None detected.")
                continue
            for item in items:
                conf = item.get("confidence")
                badge = confidence_badge(conf)
                if field_name == "skills":
                    yrs = item.get("years_experience")
                    yrs_str = f"{yrs} yrs" if yrs is not None else "years: unresolved"
                    st.write(f"- {badge}  **{item.get('name')}** — {yrs_str}")
                elif field_name == "education":
                    year = item.get("graduation_year")
                    year_str = year if year is not None else "year: unresolved"
                    st.write(f"- {badge}  {item.get('degree')}, {item.get('institution')} ({year_str})")
                elif field_name == "certifications":
                    issuer = item.get("issuer") or "issuer: unresolved"
                    st.write(f"- {badge}  {item.get('name')} — {issuer}")
                elif field_name == "career_gaps":
                    st.write(f"- {badge}  {item.get('start_year')}–{item.get('end_year')} "
                             f"({item.get('duration_months')} months)")
                with st.expander("evidence"):
                    st.caption(item.get("evidence_span") or "(no evidence span — flagged low confidence)")


input_text = st.text_area("Resume / cover-letter text", height=280, placeholder="Paste text here...")
uploaded = st.file_uploader("...or upload a .txt file", type=["txt"])
if uploaded is not None:
    input_text = uploaded.read().decode("utf-8", errors="ignore")

run = st.button("Run extraction", type="primary", disabled=not input_text)

if run:
    baseline_model, baseline_tok = get_baseline()
    ft_model, ft_tok = get_fine_tuned()

    col1, col2 = st.columns(2)

    with st.spinner("Running baseline (prompt-only)..."):
        baseline_pred, baseline_raw = run_extraction(baseline_model, baseline_tok, input_text)
    render_extraction(col1, "Baseline (prompt-only, zero-shot)", baseline_pred, baseline_raw)

    if ft_model is not None:
        with st.spinner("Running fine-tuned model..."):
            ft_pred, ft_raw = run_extraction_ft(ft_model, ft_tok, input_text)
        render_extraction(col2, "Fine-Tuned (LoRA adapter)", ft_pred, ft_raw)
    else:
        with col2:
            st.subheader("Fine-Tuned (LoRA adapter)")
            st.warning(
                f"No adapter found at `{ADAPTER_DIR}`. Run `src/fine_tune.py` first, "
                "or point ADAPTER_DIR at your trained adapter."
            )

    with st.expander("Raw JSON — baseline"):
        st.code(json.dumps(baseline_pred, indent=2) if baseline_pred else str(baseline_raw))
    if ft_model is not None:
        with st.expander("Raw JSON — fine-tuned"):
            st.code(json.dumps(ft_pred, indent=2) if ft_pred else str(ft_raw))
