"""
Shared model loading / generation utilities used by the Streamlit app and the
evaluation script, so that base-model and fine-tuned-model inference is done
identically in both places (same prompt construction, same decoding params) --
this is what makes the side-by-side comparison an apples-to-apples one.
"""
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train.prepare_training_data import SYSTEM_PROMPT, build_user_turn
from rag.retriever import PolicyRetriever

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = Path(__file__).parent.parent / "train" / "output" / "bytehive-lora"

GEN_KWARGS = dict(
    max_new_tokens=300,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.1,
)


class BytehiveReplyGenerator:
    """Loads the base model once, and optionally attaches/detaches the LoRA
    adapter so both base and fine-tuned generation share a single set of
    frozen weights in memory (important for the "realistic resource budget"
    constraint -- we do not load two separate 7B models)."""

    def __init__(self, adapter_dir: Path = ADAPTER_DIR, load_in_4bit: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        self.has_adapter = adapter_dir.exists()
        if self.has_adapter:
            self.tuned_model = PeftModel.from_pretrained(self.base_model, str(adapter_dir))
        else:
            self.tuned_model = None

        self.retriever = PolicyRetriever()

    def _generate(self, model, ticket: str, context: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_turn(ticket, context)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **GEN_KWARGS,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def retrieve_context(self, ticket: str, k: int = 3):
        return self.retriever.retrieve_with_uncertainty(ticket, k=k)

    def generate_base(self, ticket: str, context: str) -> str:
        return self._generate(self.base_model, ticket, context)

    def generate_finetuned(self, ticket: str, context: str) -> str:
        if not self.has_adapter:
            raise RuntimeError(
                f"No fine-tuned adapter found at {ADAPTER_DIR}. Run train/finetune_lora.py first, "
                "or point ADAPTER_DIR at your trained adapter."
            )
        return self._generate(self.tuned_model, ticket, context)

    def compare(self, ticket: str, k: int = 3):
        """Runs retrieval once, then generates from both models on identical
        (ticket, context) input. Returns a dict ready for the Streamlit UI."""
        chunks, is_uncertain = self.retrieve_context(ticket, k=k)
        context = self.retriever.format_context(chunks)

        base_reply = self.generate_base(ticket, context)
        tuned_reply = self.generate_finetuned(ticket, context) if self.has_adapter else None

        return {
            "ticket": ticket,
            "context": context,
            "chunks": chunks,
            "is_uncertain": is_uncertain,
            "base_reply": base_reply,
            "tuned_reply": tuned_reply,
        }
