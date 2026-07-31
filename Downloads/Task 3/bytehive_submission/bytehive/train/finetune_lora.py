"""
QLoRA fine-tuning of a top-performing, freely-fine-tunable open base model on
ByteHive's tone-and-policy-grounded support replies.

Base model: Qwen2.5-7B-Instruct (Apache 2.0, open weights, freely fine-tunable --
satisfies the "no paid-only / closed-license base model" constraint). Swap
BASE_MODEL below for "meta-llama/Llama-3.1-8B-Instruct" if you prefer Llama (gated
but free, requires accepting the license on Hugging Face) -- both fit the same
QLoRA recipe below.

Resource budget: 4-bit QLoRA on a 7-8B model fits on a single 16GB GPU (e.g. a
free/cheap Colab T4/L4, or a single consumer 4060 Ti 16GB / 3090 24GB). This is the
realistic-student-budget path called for in the brief; full fine-tuning of a 7-8B
model is NOT attempted here on purpose.

Usage:
    python train/prepare_training_data.py      # run first
    python train/finetune_lora.py

Output:
    train/output/bytehive-lora/   -- LoRA adapter weights (small, a few hundred MB)
"""
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / "data" / "train.jsonl"
EVAL_PATH = ROOT / "data" / "eval.jsonl"
OUTPUT_DIR = ROOT / "output" / "bytehive-lora"

MAX_SEQ_LEN = 1024


def load_chat_dataset(path: Path, tokenizer) -> Dataset:
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n")]
    texts = [
        tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        for r in records
    ]
    return Dataset.from_dict({"text": texts})


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_chat_dataset(TRAIN_PATH, tokenizer)
    eval_ds = load_chat_dataset(EVAL_PATH, tokenizer)
    print(f"Loaded {len(train_ds)} train / {len(eval_ds)} eval examples")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Attention + MLP projections -- standard, effective coverage for
        # instruction/style fine-tuning without touching embeddings/lm_head
        # (keeps the adapter small and training stable on modest data).
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,   # effective batch size 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
