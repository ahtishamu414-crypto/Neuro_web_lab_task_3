"""
Evaluation report generator.

For each ticket in the eval split, generates a reply from BOTH the base model and
the fine-tuned model (identical retrieved context for both), then scores each reply
on two independent axes:

  1. Tone consistency  (eval/judge_prompts.TONE_JUDGE_PROMPT)
  2. Policy grounding   (claim extraction + per-claim entailment check against the
                          retrieved context, via CLAIM_EXTRACTION_PROMPT /
                          GROUNDING_JUDGE_PROMPT)

The judge model is the *base* Qwen2.5-7B-Instruct model itself (not the fine-tuned
adapter), loaded once and reused for both generation and judging, to stay within a
single-model resource budget. This is a known limitation (self-evaluation bias) --
documented in WRITEUP.md, along with the recommended upgrade path (an external judge
model or a small set of human-graded examples used to validate the automated scores).

Usage:
    python eval/evaluate.py --n 20

Output:
    eval/report/eval_report.json   -- raw per-example scores
    eval/report/eval_summary.csv   -- aggregate table (this is what goes in the
                                       submission's evaluation report)
"""
import argparse
import csv
import json
import re
from pathlib import Path

import torch

from app.model_utils import BytehiveReplyGenerator
from eval.judge_prompts import (
    TONE_JUDGE_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
    GROUNDING_JUDGE_PROMPT,
)

ROOT = Path(__file__).parent.parent
EVAL_DATA_PATH = ROOT / "train" / "data" / "eval.jsonl"
TONE_GUIDE_PATH = ROOT / "data" / "tone_guide.md"
REPORT_DIR = Path(__file__).parent / "report"


def _extract_json(text: str):
    """LLM judges sometimes wrap JSON in prose/backticks; pull out the first
    {...} or [...] block and parse it, falling back to None on failure."""
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


class Judge:
    """Wraps the base model for judging calls (greedy, low temperature -- judging
    should be as deterministic as practical, unlike the creative generation calls)."""

    def __init__(self, generator: BytehiveReplyGenerator):
        self.tokenizer = generator.tokenizer
        self.model = generator.base_model  # judge with the untuned base model

    def _ask(self, prompt: str, max_new_tokens: int = 300) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def score_tone(self, reply: str, tone_guide: str) -> dict:
        prompt = TONE_JUDGE_PROMPT.format(tone_guide=tone_guide, reply=reply)
        raw = self._ask(prompt)
        parsed = _extract_json(raw) or {}
        dims = ["directness", "friendliness", "specificity", "no_boilerplate", "conciseness"]
        scores = {d: parsed.get(d, None) for d in dims}
        valid = [v for v in scores.values() if isinstance(v, (int, float))]
        scores["tone_avg"] = sum(valid) / len(valid) if valid else None
        scores["rationale"] = parsed.get("rationale", "")
        return scores

    def extract_claims(self, reply: str) -> list[str]:
        prompt = CLAIM_EXTRACTION_PROMPT.format(reply=reply)
        raw = self._ask(prompt)
        parsed = _extract_json(raw)
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
        return []

    def score_grounding(self, reply: str, context: str) -> dict:
        claims = self.extract_claims(reply)
        labels = []
        for claim in claims:
            prompt = GROUNDING_JUDGE_PROMPT.format(context=context, claim=claim)
            raw = self._ask(prompt, max_new_tokens=120)
            parsed = _extract_json(raw) or {}
            labels.append({
                "claim": claim,
                "label": parsed.get("label", "unsupported"),
                "reason": parsed.get("reason", ""),
            })
        n = len(labels)
        n_supported = sum(1 for l in labels if l["label"] == "supported")
        n_unsupported = sum(1 for l in labels if l["label"] == "unsupported")
        n_contradicted = sum(1 for l in labels if l["label"] == "contradicted")
        grounding_rate = (n_supported / n) if n > 0 else 1.0  # no claims = nothing to hallucinate
        return {
            "claims": labels,
            "n_claims": n,
            "n_supported": n_supported,
            "n_unsupported": n_unsupported,
            "n_contradicted": n_contradicted,
            "grounding_rate": grounding_rate,
            "has_hallucination": (n_unsupported + n_contradicted) > 0,
        }


def run_eval(n: int | None, k: int = 3):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tone_guide = TONE_GUIDE_PATH.read_text(encoding="utf-8")

    records = [json.loads(l) for l in EVAL_DATA_PATH.read_text(encoding="utf-8").strip().split("\n")]
    if n:
        records = records[:n]

    print("Loading models (base + adapter)...")
    generator = BytehiveReplyGenerator()
    judge = Judge(generator)

    results = []
    for i, rec in enumerate(records):
        # eval.jsonl stores chat-formatted messages; pull ticket/context back out
        user_content = rec["messages"][1]["content"]
        ticket = user_content.split("Customer ticket:\n", 1)[-1]
        context = user_content.split("Policy context:\n", 1)[-1].split("\n\nCustomer ticket:")[0]

        print(f"[{i+1}/{len(records)}] {rec['category']}")

        base_reply = generator.generate_base(ticket, context)
        tone_base = judge.score_tone(base_reply, tone_guide)
        grounding_base = judge.score_grounding(base_reply, context)

        row = {
            "category": rec["category"],
            "uncertain_expected": rec["uncertain"],
            "ticket": ticket,
            "context": context,
            "base_reply": base_reply,
            "base_tone": tone_base,
            "base_grounding": grounding_base,
        }

        if generator.has_adapter:
            tuned_reply = generator.generate_finetuned(ticket, context)
            tone_tuned = judge.score_tone(tuned_reply, tone_guide)
            grounding_tuned = judge.score_grounding(tuned_reply, context)
            row.update({
                "tuned_reply": tuned_reply,
                "tuned_tone": tone_tuned,
                "tuned_grounding": grounding_tuned,
            })

        results.append(row)

    with (REPORT_DIR / "eval_report.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    write_summary_csv(results)
    print(f"\nSaved detailed report to {REPORT_DIR / 'eval_report.json'}")
    print(f"Saved summary table to {REPORT_DIR / 'eval_summary.csv'}")


def write_summary_csv(results):
    has_tuned = any("tuned_reply" in r for r in results)
    fieldnames = [
        "model", "n_examples", "avg_tone_directness", "avg_tone_friendliness",
        "avg_tone_specificity", "avg_tone_no_boilerplate", "avg_tone_conciseness",
        "avg_tone_overall", "avg_grounding_rate", "pct_replies_with_hallucination",
    ]

    def summarize(key_prefix: str, model_name: str):
        tones = [r[f"{key_prefix}_tone"] for r in results if r[f"{key_prefix}_tone"].get("tone_avg") is not None]
        groundings = [r[f"{key_prefix}_grounding"] for r in results]
        dims = ["directness", "friendliness", "specificity", "no_boilerplate", "conciseness"]

        def avg(vals):
            vals = [v for v in vals if isinstance(v, (int, float))]
            return round(sum(vals) / len(vals), 2) if vals else None

        row = {"model": model_name, "n_examples": len(results)}
        for d in dims:
            row[f"avg_tone_{d}"] = avg([t.get(d) for t in tones])
        row["avg_tone_overall"] = avg([t.get("tone_avg") for t in tones])
        row["avg_grounding_rate"] = avg([g["grounding_rate"] for g in groundings])
        n_hallucinated = sum(1 for g in groundings if g["has_hallucination"])
        row["pct_replies_with_hallucination"] = round(100 * n_hallucinated / len(groundings), 1) if groundings else None
        return row

    rows = [summarize("base", "base_model")]
    if has_tuned:
        rows.append(summarize("tuned", "fine_tuned_model"))

    with (REPORT_DIR / "eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Limit number of eval examples (default: all)")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved chunks per query")
    args = parser.parse_args()
    run_eval(n=args.n, k=args.k)
