# Evaluation Report — Baseline vs. Fine-Tuned Extraction

*Fill in the bracketed values by running the reproduction steps in
`README.md` §5 on a GPU environment, then pasting the JSON output of
`src/evaluate.py` into the tables below.*

## Setup

- Base model (both baseline and fine-tuned): `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tuning method: LoRA, r=16, alpha=32, 3 epochs (full hyperparameters
  in `README.md` §3)
- Test set: `data/test.jsonl` — [N] examples, in-distribution industries
- OOD test set: `data/test_ood.jsonl` — [N] examples, industries (marketing,
  construction) excluded from training entirely

## Results — in-distribution test set

| Field | Baseline P / R / F1 | Fine-Tuned P / R / F1 | Δ F1 |
|---|---|---|---|
| Skills | [ ] / [ ] / [ ] | [ ] / [ ] / [ ] | [ ] |
| Education | [ ] / [ ] / [ ] | [ ] / [ ] / [ ] | [ ] |
| Certifications | [ ] / [ ] / [ ] | [ ] / [ ] / [ ] | [ ] |
| Career Gaps | [ ] / [ ] / [ ] | [ ] / [ ] / [ ] | [ ] |
| **Macro F1** | **[ ]** | **[ ]** | **[ ]** |

| Metric | Baseline | Fine-Tuned |
|---|---|---|
| JSON parse failure rate | [ ] | [ ] |
| Hallucination rate | [ ] | [ ] |
| Null-precision (correctly left unresolved) | [ ] | [ ] |

## Results — out-of-distribution test set (unseen industries)

*(same table shape, run against `data/test_ood.jsonl` predictions)*

| Field | Baseline P / R / F1 | Fine-Tuned P / R / F1 | Δ F1 |
|---|---|---|---|
| Skills | [ ] | [ ] | [ ] |
| Education | [ ] | [ ] | [ ] |
| Certifications | [ ] | [ ] | [ ] |
| Career Gaps | [ ] | [ ] | [ ] |
| **Macro F1** | **[ ]** | **[ ]** | **[ ]** |

## Interpretation checklist

- [ ] Does fine-tuned macro F1 clearly exceed baseline on the in-distribution
  test set? (This is the client's core success criterion.)
- [ ] Does the fine-tuned model's OOD macro F1 hold up reasonably close to
  its in-distribution macro F1 (small gap = generalizes; large gap = still
  overfitting to training industries, revisit LoRA rank / epochs / data mix)?
- [ ] Is the fine-tuned hallucination rate lower than the baseline's, not
  just similar? (A fine-tune that matches baseline hallucination rate but
  wins on F1 is still a rollout risk for the "never fabricate a field"
  requirement.)
- [ ] Is null-precision for the fine-tuned model meaningfully higher than
  baseline? This is the single number that most directly answers "did we
  fix the guessing problem."

## Cost comparison (fill in from your run)

| | Current (hosted general-purpose model, prompt-only) | Fine-tuned (self-hosted) |
|---|---|---|
| Per-application inference cost | [TalentGrid's current hosted API cost] | [ ] (GPU-hour amortized) |
| Weekly manual QA correction volume | [current %] | [projected — should track null-precision improvement] |
