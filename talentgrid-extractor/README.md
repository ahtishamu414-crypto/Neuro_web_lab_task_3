# TalentGrid Structured Extraction — Fine-Tuned Model

Client Problem 4: fine-tune an open text model to replace TalentGrid's
prompt-only extraction pipeline, with a rigorous baseline comparison, an
anti-hallucination design, and a Streamlit side-by-side demo.

## 1. Approach

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` — open-weight, Apache 2.0
license, no paid tier required to fine-tune. 1.5B parameters keeps LoRA
fine-tuning and inference inside a single consumer GPU / free-tier Colab
budget. (Swap to `Qwen2.5-7B-Instruct` in `BASE_MODEL` if more GPU is
available — nothing else in the pipeline changes.)

**Method:** LoRA (via PEFT + TRL's `SFTTrainer`), not full fine-tuning.
Full fine-tuning on a narrow task is what caused catastrophic forgetting in
the prior attempt; LoRA freezes the base weights and trains small adapters,
which preserves general language ability while still learning the schema.

**Fixed schema:** skills (name, years_experience, confidence,
evidence_span), education, certifications, career_gaps — see
`data/schema.py` for the full JSON schema and system prompt used by *both*
the baseline and the fine-tune, so the only variable between the two is
whether the adapter is attached.

**Anti-fabrication design (the client's core ask):**
- Every field must carry an `evidence_span` — a verbatim substring of the
  source text. A field with no textual support gets `null` + low
  `confidence`, never a guess.
- Training data was generated so that a meaningful fraction of every field
  has **no** stated value (e.g. a skill mentioned with no years, a cert
  with no issuer) — the model only ever sees "no evidence → null" as the
  correct target, never "no evidence → plausible guess."
- Evaluation checks this automatically: any predicted `evidence_span` that
  isn't actually a substring of the input, or any predicted entity with no
  ground-truth counterpart, counts as a hallucination (see
  `hallucination_rate` and `null_precision` in `src/evaluate.py`).

**Overfitting / generalization design:**
- Synthetic resumes span 8 in-distribution industries (software, nursing,
  accounting, logistics, teaching, sales, manufacturing, legal) across 4
  distinct formats (bullet list, narrative cover letter, dense paragraph,
  chronological table).
- Two industries (marketing, construction) are **entirely excluded** from
  train/val and only appear in `data/test_ood.jsonl`, so the evaluation can
  directly measure whether fine-tuning improved or hurt generalization to
  unseen domains — the exact comparison the client's post-mortem says was
  skipped last time.

## 2. Why synthetic data

TalentGrid's real applicant data is real candidates' PII and isn't
available for a student project. `data/generate_synthetic_data.py`
generates resumes with perfect, checkable ground truth (including
deliberately ambiguous/missing fields) across many industries and formats.
**Before running this on TalentGrid's actual data**, swap in their labeled
sample in the same JSONL shape (`{"text": ..., "label": {...}}`) — nothing
else in the pipeline needs to change.

## 3. Hyperparameters (see `src/fine_tune.py` for the source of truth)

| Hyperparameter | Value | Rationale |
|---|---|---|
| LoRA rank (r) | 16 | Standard starting point for a 1.5B model; enough capacity for a fixed-schema task without approaching full fine-tune expressiveness |
| LoRA alpha | 32 | 2x rank, standard scaling |
| LoRA dropout | 0.05 | Light regularization against overfitting on synthetic data |
| Target modules | q/k/v/o projections | Attention-only adapters are the common cost/quality sweet spot for instruction-following tasks |
| Learning rate | 2e-4 | Typical LoRA LR range (higher than full fine-tuning LR) |
| Epochs | 3 | Deliberately capped — more epochs on a narrow synthetic set is how the prior attempt overfit; combined with `load_best_model_at_end` on eval loss |
| Effective batch size | 16 (2 × 8 grad-accum) | Fits a 16 GB GPU at `MAX_SEQ_LEN=1536` |

## 4. Resource usage (expected, single consumer GPU e.g. RTX 4060 16GB / Colab T4)

- Dataset: ~336 train / 72 val / 72 test / 120 OOD-test examples (from the
  default `n_per_industry=60`; scale up via the `--n_per_industry` style
  argument if you extend the generator).
- LoRA fine-tune: ~15–30 minutes for 3 epochs at this dataset size and
  sequence length.
- Adapter size on disk: tens of MB (vs. multiple GB for a full fine-tune
  checkpoint).
- Inference: a few seconds per resume on GPU, both baseline and fine-tuned,
  since both share the same 1.5B base weights loaded once.

## 5. Reproduction steps (clean environment)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Generate the synthetic dataset (or substitute TalentGrid's real labeled data)
python data/generate_synthetic_data.py

# 2. Run the baseline (prompt-only) on held-out splits
python src/baseline.py --input data/test.jsonl     --output outputs/baseline_test.jsonl
python src/baseline.py --input data/test_ood.jsonl --output outputs/baseline_ood.jsonl

# 3. Fine-tune
python src/fine_tune.py --train data/train.jsonl --val data/val.jsonl \
    --output_dir adapters/qwen2.5-1.5b-extraction-lora

# 4. Run the fine-tuned model on the same held-out splits
python -c "
from src.fine_tuned_inference import load_fine_tuned, run_extraction_ft
import json
model, tok = load_fine_tuned('adapters/qwen2.5-1.5b-extraction-lora')
for split in ['test', 'test_ood']:
    with open(f'data/{split}.jsonl') as fin, open(f'outputs/finetuned_{split}.jsonl', 'w') as fout:
        for line in fin:
            ex = json.loads(line)
            pred, raw = run_extraction_ft(model, tok, ex['text'])
            fout.write(json.dumps({'text': ex['text'], 'label': ex['label'], 'prediction': pred}) + '\n')
"

# 5. Evaluate baseline vs fine-tuned, on both in-distribution and OOD splits
python src/evaluate.py --baseline_preds outputs/baseline_test.jsonl \
    --ft_preds outputs/finetuned_test.jsonl --split_name test --output reports/results_test.json
python src/evaluate.py --baseline_preds outputs/baseline_ood.jsonl \
    --ft_preds outputs/finetuned_ood.jsonl --split_name test_ood --output reports/results_ood.json

# 6. Launch the demo app
streamlit run app/streamlit_app.py
```

## 6. What's actually included vs. what requires a GPU you run yourself

This submission includes, complete and tested:
- The full fixed schema and shared system prompt (`data/schema.py`)
- A synthetic data generator, run and verified to produce train/val/test/OOD
  splits (`data/generate_synthetic_data.py`)
- Baseline, fine-tuning, and fine-tuned-inference code
  (`src/baseline.py`, `src/fine_tune.py`, `src/fine_tuned_inference.py`)
- The evaluation harness, **smoke-tested against synthetic predictions** to
  confirm the precision/recall/F1, hallucination-rate, and null-precision
  logic is correct (`src/evaluate.py`)
- The Streamlit side-by-side comparison app (`app/streamlit_app.py`),
  which runs in "baseline only" mode until an adapter is present, so it's
  demoable at every stage

**Not included:** actual trained adapter weights and the resulting
head-to-head numbers on `data/test.jsonl` / `data/test_ood.jsonl`. Model
fine-tuning needs a GPU and a download of the base model's weights from the
Hugging Face Hub, neither of which is available in this authoring
environment (no GPU, and the sandbox's network allowlist doesn't include
`huggingface.co`). `reports/evaluation_report_template.md` shows exactly
what the final report will contain — run steps 2–5 above on your own
GPU/Colab to fill in the real numbers; every script needed to produce them
end-to-end is here and has been syntax-checked and logic-tested with
synthetic predictions.

## 7. Known limitations

- Synthetic training/eval data is a proxy for TalentGrid's real applicant
  distribution; real resumes will have messier formatting (OCR artifacts
  from scanned PDFs, non-English sections, inconsistent date formats) that
  this generator doesn't fully model. Expect a first real-data pass to
  surface new failure formats — recommend a second, smaller fine-tuning
  round on a real labeled sample once available, before wider rollout.
- Evidence-span hallucination checking is a substring match — a model that
  paraphrases genuinely-supported text (rather than copying it verbatim)
  will be flagged as a false hallucination. In practice this is a
  conservative bias (favors precision on the hallucination metric), but
  worth knowing when reading the numbers.
- LoRA target modules are attention-only; if evaluation numbers show
  underfitting relative to budget, widening `LORA_TARGET_MODULES` to include
  MLP projections is the next lever to pull before moving to a larger base
  model.
