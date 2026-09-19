"""Language modeling dataset."""
from __future__ import annotations
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer

class LanguageModelingDataset(Dataset):
    def __init__(self, seq_len=2048, tokenizer_name="gpt2",
                 max_tokens=1_000_000, use_synthetic=True):
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if use_synthetic:
            self.tokens = torch.randint(0, self.tokenizer.vocab_size, (max_tokens,))
        else:
            try:
                from datasets import load_dataset
                ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
                texts = "\n\n".join(ds["text"][:10000])
                enc = self.tokenizer(texts, return_tensors="pt", truncation=False)
                self.tokens = enc["input_ids"].squeeze(0)[:max_tokens]
            except Exception:
                self.tokens = torch.randint(0, self.tokenizer.vocab_size, (max_tokens,))
        self.chunks = [self.tokens[i:i+seq_len+1]
                       for i in range(0, len(self.tokens)-seq_len-1, seq_len//2)]

    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        c = self.chunks[idx]
        return {"input_ids": c[:-1], "labels": c[1:]}

    @property
    def vocab_size(self): return self.tokenizer.vocab_size

def create_lm_dataloaders(seq_len=2048, batch_size=16, val_split=0.05,
                          num_workers=4, use_synthetic=True):
    ds = LanguageModelingDataset(seq_len=seq_len, use_synthetic=use_synthetic)
    val_size = int(len(ds)*val_split)
    train_ds, val_ds = random_split(ds, [len(ds)-val_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs))
