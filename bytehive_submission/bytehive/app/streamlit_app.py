"""
ByteHive Support Draft Assistant -- Streamlit app.

Run with:
    streamlit run app/streamlit_app.py

An agent pastes a ticket, retrieves the grounding policy context automatically (RAG),
and gets a drafted reply from the fine-tuned model side-by-side with the untuned base
model's draft on the identical ticket + context, so the team can see exactly what the
fine-tuning changed. If retrieval doesn't find a confident policy match, the app shows
an uncertainty banner instead of letting either model's draft look more authoritative
than it actually is.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))  # allow `rag.`, `train.` imports

from app.model_utils import BytehiveReplyGenerator  # noqa: E402

st.set_page_config(page_title="ByteHive Support Draft Assistant", layout="wide")


@st.cache_resource(show_spinner="Loading models (this happens once)...")
def load_generator():
    return BytehiveReplyGenerator()


def render_context(chunks):
    if not chunks:
        st.caption("No policy chunks retrieved.")
        return
    for c in chunks:
        with st.expander(f"{c.source} \u2014 {c.header}  (similarity {c.score:.2f})"):
            st.markdown(c.text)


def main():
    st.title("ByteHive Support Draft Assistant")
    st.caption(
        "Paste a ticket to get a drafted reply. Compares the fine-tuned, "
        "policy-grounded model against the untuned base model on the same ticket."
    )

    try:
        generator = load_generator()
    except Exception as e:
        st.error(
            "Couldn't load the model(s). If you haven't fine-tuned yet, base-model-only "
            f"comparison will still work once the base model loads. Details: {e}"
        )
        return

    if not generator.has_adapter:
        st.warning(
            "No fine-tuned adapter found yet -- showing base model only. "
            "Run `python train/finetune_lora.py` to train the adapter, then restart the app."
        )

    ticket = st.text_area(
        "Customer ticket",
        height=140,
        placeholder="Paste the customer's message here, e.g. 'I was charged twice this month for the Growth plan...'",
    )
    k = st.slider("Policy chunks to retrieve", min_value=1, max_value=5, value=3)
    run = st.button("Generate draft reply", type="primary", disabled=not ticket.strip())

    if run:
        with st.spinner("Retrieving policy context and generating drafts..."):
            result = generator.compare(ticket, k=k)

        if result["is_uncertain"]:
            st.warning(
                "⚠️ **Low-confidence policy match.** Retrieval did not find a policy "
                "section that clearly covers this ticket. Treat both drafts below as "
                "provisional -- verify against the knowledge base or escalate before sending."
            )

        st.subheader("Retrieved policy context")
        render_context(result["chunks"])

        st.subheader("Drafted replies")
        col_base, col_tuned = st.columns(2)

        with col_base:
            st.markdown("##### Base model (untuned)")
            st.info(result["base_reply"])

        with col_tuned:
            st.markdown("##### Fine-tuned model (ByteHive tone + grounded)")
            if result["tuned_reply"] is not None:
                st.success(result["tuned_reply"])
            else:
                st.caption("Fine-tuned adapter not loaded -- train it first.")

        st.divider()
        st.caption(
            "Both drafts were generated from the exact same retrieved policy context above. "
            "Differences you see reflect tone and grounding behavior learned during fine-tuning, "
            "not different information being shown to each model."
        )

    with st.sidebar:
        st.header("About this app")
        st.markdown(
            "- Retrieval-augmented: policy facts come from `data/policies/*.md` via a "
            "FAISS index, not from the model's memorized weights.\n"
            "- Fine-tuning shapes tone/structure only, per ByteHive's voice guide.\n"
            "- See `eval/report/eval_summary.csv` for tone-vs-grounding scores across "
            "the evaluation set.\n"
        )
        st.markdown("---")
        st.caption("Internal tool -- not customer-facing.")


if __name__ == "__main__":
    main()
