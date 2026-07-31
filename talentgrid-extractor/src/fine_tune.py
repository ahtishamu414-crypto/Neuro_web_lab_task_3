"""
LoRA fine-tuning of Qwen2.5-1.5B-Instruct for structured resume extraction.

WHY THIS MODEL
- Open-weight (Apache 2.0), genuinely free to fine-tune -- satisfies the
  client's licensing constraint.
- 1.5B parameters: a LoRA fine-tune fits comfortably on a single consumer
  GPU (e.g. a 16 GB T4/RTX 4060) or Colab's free tier, satisfying the
  "realistic resource budget" constraint. (Qwen2.5-7B-Instruct is a drop-in
  swap -- change BASE_MODEL below -- if more GPU budget is available; the
  8B/9B tier of Llama-3.1 or Gemma-2 are other valid open swaps.)

WHY LoRA, NOT FULL FINE-TUNING
Full fine-tuning on a narrow task is exactly what caused "Catastrophic
Forgetting" in the previous attempt -- it overwrites the base model's
general language weights. LoRA freezes the base weights and trains small
rank-decomposition adapters on top, which empirically preserves general
capability far better while still learning the task. This directly targets
that failure mode, not just cost.

WHY THE TRAINING TARGETS LOOK THE WAY THEY DO
Every training example's target JSON was generated with explicit nulls for
fields the source text doesn't support (see data/generate_synthetic_data.py).
The model is therefore never shown a single example where guessing a missing
value was rewarded -- this is what directly targets "Schema Drift" (invented
plausible-but-wrong values). We also cap epochs and use a held-out val loss
for early stopping specifically to keep the adapter from overfitting to the
in-distribution industries (see test_ood.jsonl in evaluate.py).

Usage:
    python src/fine_tune.py --train data/train.jsonl --val data/val.jsonl \
        --output_dir adapters/qwen2.5-1.5b-extraction-lora
"""
import argparse
import json

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
import torch

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.schema import SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Hyperparameters -- chosen for a single consumer GPU (~16 GB) budget.
# Documented here (and repeated in the write-up) so the run is reproducible.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3            # kept low deliberately -- more epochs on a narrow
                          # synthetic set is how the previous attempt overfit
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8      # effective batch size 16
MAX_SEQ_LEN = 1536


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            rows.append(ex)
    return rows


def to_chat_example(ex, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex["text"]},
        {"role": "assistant", "content": json.dumps(ex["label"], ensure_ascii=False)},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def build_dataset(path, tokenizer):
    rows = load_jsonl(path)
    ds = Dataset.from_list(rows)
    ds = ds.map(lambda ex: to_chat_example(ex, tokenizer), remove_columns=ds.column_names)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--output_dir", default="adapters/qwen2.5-1.5b-extraction-lora")
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = build_dataset(args.train, tokenizer)
    val_ds = build_dataset(args.val, tokenizer)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=MAX_SEQ_LEN,
        bf16=torch.cuda.is_available(),
        report_to=[],
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
