"""Inference for PEFT fine-tuned LLM."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.utils.logger import get_logger
log = get_logger(__name__)

class PEFTPredictor:
    def __init__(self, base_model_name, adapter_path: Union[str,Path],
                 device=None, max_new_tokens=512, temperature=0.7):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._load(base_model_name, Path(adapter_path))

    def _load(self, base_model_name, adapter_path):
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = PeftModel.from_pretrained(base, str(adapter_path))
        self.model.eval()

    def _fmt_prompt(self, instruction, input_text=""):
        if input_text.strip():
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    @torch.inference_mode()
    def predict_single(self, instruction, input_text="") -> dict:
        prompt = self._fmt_prompt(instruction, input_text)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        outputs = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, do_sample=True, top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id)
        generated = outputs[0][prompt_len:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {"instruction":instruction,"input":input_text,"response":response,
                "num_tokens_generated":len(generated)}

    @torch.inference_mode()
    def predict_batch(self, instructions, inputs=None) -> list[dict]:
        if inputs is None: inputs = [""]*len(instructions)
        return [self.predict_single(i, inp) for i,inp in zip(instructions, inputs)]
