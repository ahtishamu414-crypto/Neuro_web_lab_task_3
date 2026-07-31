"""
Prompt-only baseline: this stands in for TalentGrid's *current* production
pipeline -- a general-purpose hosted model called with a hand-tuned prompt,
zero-shot, no fine-tuning. We reuse the exact same base model weights as the
fine-tune (see fine_tune.py) so the evaluation isolates the effect of
fine-tuning itself, rather than confounding it with a different base model
-- that confound is exactly what "No Real Baseline Comparison" got wrong
last time.

Usage:
    python src/baseline.py --input data/test.jsonl --output outputs/baseline_test.jsonl
"""
import argparse
import json
import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.schema import SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # open-weight, Apache-2.0, free to fine-tune


def load_model(model_name=BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tok


def extract_json(raw_text: str):
    """Best-effort recovery of the JSON object from a raw generation, since
    an un-fine-tuned model will sometimes wrap it in prose or code fences."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def run_extraction(model, tok, resume_text: str, max_new_tokens=800):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": resume_text},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None)
    generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    parsed = extract_json(generated)
    return parsed, generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=BASE_MODEL)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            ex = json.loads(line)
            parsed, raw = run_extraction(model, tok, ex["text"])
            fout.write(json.dumps({
                "text": ex["text"], "label": ex["label"],
                "prediction": parsed, "raw_generation": raw,
            }) + "\n")


if __name__ == "__main__":
    main()
