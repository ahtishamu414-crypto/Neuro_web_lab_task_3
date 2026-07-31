"""
Converts data/tickets_dataset.jsonl into chat-formatted training examples for LoRA
fine-tuning, and splits into train/eval sets.

Design choice: the *ticket + retrieved policy context* pair defines the input, and
the *tone-correct, policy-grounded reply* is the target. This is deliberate: it
teaches the model to (a) write in ByteHive's voice and (b) restrict its factual
claims to what's in the provided context -- rather than teaching it to memorize
policy facts directly into its weights, which is what caused the "style overfit,
substance lost" failure mode in the earlier attempt (see WRITEUP.md).

Usage:
    python train/prepare_training_data.py
Produces:
    train/data/train.jsonl
    train/data/eval.jsonl
Each line is: {"messages": [{"role": "system", ...}, {"role": "user", ...},
                             {"role": "assistant", ...}]}
ready for trl's SFTTrainer with a chat template.
"""
import json
import random
from pathlib import Path

SEED = 13
EVAL_FRACTION = 0.15

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "data" / "tickets_dataset.jsonl"
TONE_GUIDE_PATH = ROOT / "data" / "tone_guide.md"
OUT_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """You are a ByteHive customer support agent. Write replies that are:
- Direct: lead with the answer/outcome in the first sentence.
- Friendly, not formal: contractions are fine, no "Dear Valued Customer" style boilerplate.
- Specific: use exact numbers/dates from the policy context, not vague words like "shortly".
- Short paragraphs (1-3 sentences).
- Grounded ONLY in the "Policy context" provided below. Never state a refund, billing, \
or cancellation rule that is not explicitly supported by that context.
- If the policy context does not clearly cover the customer's question, say so plainly \
and offer to check/escalate rather than guessing.
End with a short, casual sign-off. Never use a formal signature block."""


def build_user_turn(ticket: str, context: str) -> str:
    return f"Policy context:\n{context}\n\nCustomer ticket:\n{ticket}"


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = DATASET_PATH.read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(l) for l in lines]
    random.shuffle(records)

    n_eval = max(1, int(len(records) * EVAL_FRACTION))
    eval_records = records[:n_eval]
    train_records = records[n_eval:]

    def to_chat(record):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_turn(record["ticket"], record["retrieved_context"])},
                {"role": "assistant", "content": record["reply"]},
            ],
            "category": record["category"],
            "uncertain": record["uncertain"],
        }

    def write(path, recs):
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")

    write(OUT_DIR / "train.jsonl", train_records)
    write(OUT_DIR / "eval.jsonl", eval_records)

    print(f"Train: {len(train_records)} examples -> {OUT_DIR / 'train.jsonl'}")
    print(f"Eval:  {len(eval_records)} examples -> {OUT_DIR / 'eval.jsonl'}")
    print(
        "\nNOTE: this seed set has 38 examples (demo scale). For the real submission, "
        "expand data/tickets_dataset.jsonl to 150-300 human-verified examples using the "
        "same schema before re-running this script -- see WRITEUP.md 'Data construction'."
    )


if __name__ == "__main__":
    main()
