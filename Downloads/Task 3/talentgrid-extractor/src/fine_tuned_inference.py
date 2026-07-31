"""
Inference wrapper for the fine-tuned adapter. Reused by evaluate.py and by
the Streamlit app so there's exactly one code path for "run the fine-tuned
model", matching the baseline's structure in src/baseline.py.
"""
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))
from data.schema import SYSTEM_PROMPT
from src.baseline import BASE_MODEL, extract_json


def load_fine_tuned(adapter_dir: str, base_model: str = BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tok


def run_extraction_ft(model, tok, resume_text: str, max_new_tokens=800):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": resume_text},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    parsed = extract_json(generated)
    return parsed, generated
