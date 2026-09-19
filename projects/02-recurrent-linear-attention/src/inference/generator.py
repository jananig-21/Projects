"""Text generation with RLA model."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import torch
from transformers import AutoTokenizer
from src.models.rla_cell import RecurrentLinearAttentionModel
from src.utils.logger import get_logger
log = get_logger(__name__)

class RLATextGenerator:
    def __init__(self, model_path: Union[str, Path], tokenizer_name="gpt2", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self._load(Path(model_path))

    def _load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        hp = ckpt["hyperparameters"]
        self.model = RecurrentLinearAttentionModel(
            vocab_size=hp["vocab_size"], hidden_dim=hp["hidden_dim"],
            num_layers=hp["num_layers"], num_heads=hp["num_heads"], max_seq_len=hp["max_seq_len"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device); self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens=200, temperature=0.8, top_k=50) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        logits, _ = self.model(input_ids)
        generated = input_ids.squeeze(0).tolist()
        current_token = torch.tensor([[generated[-1]]], device=self.device)
        for _ in range(max_new_tokens):
            logits_step, _ = self.model(current_token)
            next_logits = logits_step[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, -1:]] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token.item())
            current_token = next_token.unsqueeze(0)
            if next_token.item() == self.tokenizer.eos_token_id: break
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def predict_single(self, prompt: str, max_tokens=100) -> dict:
        generated = self.generate(prompt, max_tokens)
        return {"prompt": prompt, "generated_text": generated,
                "num_new_tokens": len(self.tokenizer(generated)["input_ids"])}

    def predict_batch(self, prompts: list[str], max_tokens=100) -> list[dict]:
        return [self.predict_single(p, max_tokens) for p in prompts]
