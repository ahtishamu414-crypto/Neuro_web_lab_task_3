# ByteHive Support Draft Assistant

Fine-tuned, retrieval-grounded support-reply drafting for ByteHive, submitted for
Internship Task 5 / Client Problem 5.

## What this is

- A **LoRA fine-tune** of `Qwen/Qwen2.5-7B-Instruct` (open weights, Apache 2.0,
  freely fine-tunable) trained to write in ByteHive's specific support tone.
- A **RAG layer** (FAISS + sentence-transformers) over ByteHive's actual refund,
  billing, and cancellation policy docs, so factual claims come from retrieved
  policy text rather than from the model's memorized weights.
- An **evaluation suite** that scores tone consistency and policy grounding
  **separately**, comparing the fine-tuned model against the untuned base model.
- A **Streamlit app** where an agent pastes a ticket and sees both models' drafts
  side-by-side, with retrieved policy context and an uncertainty flag when
  retrieval doesn't find a confident match.

See `WRITEUP.md` for the full technical write-up (approach, why RAG + LoRA instead
of fine-tuning facts into the model, data construction methodology, evaluation
methodology, and limitations).

## Project layout

```
bytehive/
├── data/
│   ├── policies/                # ByteHive's actual policy docs (refunds, billing, cancellations)
│   ├── tone_guide.md            # ByteHive brand voice guide
│   ├── build_seed_dataset.py    # generates data/tickets_dataset.jsonl
│   └── tickets_dataset.jsonl    # (ticket, retrieved_context, ideal_reply) training examples
├── rag/
│   ├── build_index.py           # chunk policies -> FAISS index
│   ├── retriever.py             # retrieval + uncertainty scoring, used by train & app
│   └── index/                   # generated: policy_index.faiss, policy_chunks.json
├── train/
│   ├── prepare_training_data.py # dataset.jsonl -> chat-formatted train/eval splits
│   ├── finetune_lora.py         # QLoRA fine-tuning script
│   ├── data/                    # generated: train.jsonl, eval.jsonl
│   └── output/bytehive-lora/    # generated: LoRA adapter weights
├── eval/
│   ├── judge_prompts.py         # tone-judge and policy-grounding-judge prompts
│   ├── evaluate.py              # runs both models on eval set, scores both axes
│   └── report/                  # generated: eval_report.json, eval_summary.csv
├── app/
│   ├── model_utils.py           # shared inference code (base + LoRA), used by app & eval
│   └── streamlit_app.py         # the Streamlit UI
├── requirements.txt
├── README.md
└── WRITEUP.md
```

## Setup

Requires Python 3.10+ and a CUDA GPU with **at least 16GB VRAM** for 4-bit QLoRA
fine-tuning (inference alone can run on less, or on CPU for the base model with
`load_in_4bit=False`, slowly). A free-tier Colab T4/L4 GPU is sufficient.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If using a gated model (e.g. swapping in Llama 3.1 instead of Qwen2.5), log in first:

```bash
huggingface-cli login
```

## Run order

```bash
# 1. Build the policy retrieval index (run once, or whenever data/policies/*.md changes)
python rag/build_index.py

# 2. Turn the seed ticket dataset into train/eval chat-format splits
python train/prepare_training_data.py

# 3. Fine-tune the LoRA adapter (needs a GPU; ~20-40 min for the seed dataset on a T4/L4)
python train/finetune_lora.py

# 4. Run the evaluation report (base vs fine-tuned, tone + grounding scores)
python eval/evaluate.py --n 20

# 5. Launch the app
streamlit run app/streamlit_app.py
```

Steps 3 and 4 require a GPU and will take longer as the dataset grows past the
38-example seed set (see `data/build_seed_dataset.py` docstring for how to scale to
150-300 examples for a stronger submission).

The app works even before step 3 has been run -- it will show the base model only
and a warning that no fine-tuned adapter was found.

## Notes on scale

The seed dataset (`data/tickets_dataset.jsonl`) ships with 38 hand-written,
policy-verified examples to keep this submission runnable end-to-end on a single
GPU within the 6-day window. It intentionally includes:
- Standard in-window and out-of-window refund cases
- Every documented exception (outage, billing error, downgrade, chargeback,
  non-refundable add-ons)
- Several "the policy docs don't clearly cover this" cases, to teach the
  uncertainty-flagging behavior
- A few "firm but polite no" cases, so the model doesn't learn to always say yes

Scaling to the full 150-300 example dataset recommended for production is a matter
of repeating the same construction process (see `WRITEUP.md`), not changing the
pipeline.
