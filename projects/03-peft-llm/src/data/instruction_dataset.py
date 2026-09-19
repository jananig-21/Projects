"""Instruction-following dataset for PEFT fine-tuning."""
from __future__ import annotations
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizer

ALPACA_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
ALPACA_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n{output}"

class InstructionDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length=512, use_synthetic=True):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []
        if use_synthetic: self._load_synthetic()
        else: self._load_alpaca()
        self._tokenize_all()

    def _load_synthetic(self):
        tasks = [
            ("Summarize the following.", "The quick brown fox.", "A fox jumps."),
            ("Translate to French.", "Hello!", "Bonjour!"),
            ("Write a Python add function.", "", "def add(a,b): return a+b"),
            ("Explain ML.", "", "ML lets models learn from data."),
            ("Fix bug: x=1/0", "", "Division by zero is invalid."),
        ]
        for instr,inp,out in tasks * 200:
            self.examples.append({"instruction":instr,"input":inp,"output":out})

    def _load_alpaca(self):
        try:
            from datasets import load_dataset
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            for ex in ds:
                self.examples.append({"instruction":ex["instruction"],"input":ex.get("input",""),"output":ex["output"]})
        except Exception: self._load_synthetic()

    def _format(self, ex):
        return ALPACA_TEMPLATE.format(**ex) if ex["input"].strip() else ALPACA_NO_INPUT.format(**ex)

    def _tokenize_all(self):
        self.tokenized = []
        for ex in self.examples:
            text = self._format(ex)
            enc = self.tokenizer(text, max_length=self.max_seq_length, truncation=True,
                                  padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attn_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            if ex["input"].strip():
                prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"
            instr_len = min(len(self.tokenizer(prompt,add_special_tokens=False)["input_ids"]), self.max_seq_length)
            labels[:instr_len] = -100
            labels[attn_mask == 0] = -100
            self.tokenized.append({"input_ids":input_ids,"attention_mask":attn_mask,"labels":labels})

    def __len__(self): return len(self.tokenized)
    def __getitem__(self, idx): return self.tokenized[idx]

def create_peft_dataloaders(tokenizer, max_seq_length=512, batch_size=8,
                             val_split=0.1, num_workers=4, use_synthetic=True):
    ds = InstructionDataset(tokenizer, max_seq_length, use_synthetic=use_synthetic)
    val_size = int(len(ds)*val_split)
    train_ds, val_ds = random_split(ds, [len(ds)-val_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs))
