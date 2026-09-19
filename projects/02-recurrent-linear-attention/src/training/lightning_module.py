"""Lightning module for RLA language model."""
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn.functional as F
import lightning as L
from src.models.rla_cell import RecurrentLinearAttentionModel

class RLALanguageModelModule(L.LightningModule):
    def __init__(self, vocab_size, hidden_dim=512, num_layers=6, num_heads=8,
                 ffn_multiplier=4, max_seq_len=2048, dropout=0.1,
                 learning_rate=3e-4, weight_decay=0.1, warmup_ratio=0.05, max_epochs=50):
        super().__init__()
        self.save_hyperparameters()
        self.model = RecurrentLinearAttentionModel(
            vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=num_heads, ffn_multiplier=ffn_multiplier,
            max_seq_len=max_seq_len, dropout=dropout)

    def forward(self, input_ids): return self.model(input_ids)

    def _loss(self, logits, labels):
        B, L, V = logits.shape
        return F.cross_entropy(logits.reshape(B*L, V), labels.reshape(B*L))

    def training_step(self, batch, batch_idx):
        logits, _ = self(batch["input_ids"])
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/ppl", torch.exp(loss), prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, _ = self(batch["input_ids"])
        loss = self._loss(logits, batch["labels"])
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/ppl",  torch.exp(loss), prog_bar=True, on_epoch=True)

    def configure_optimizers(self) -> Any:
        decay = [p for n,p in self.named_parameters() if p.ndim>=2 and p.requires_grad]
        nodecay= [p for n,p in self.named_parameters() if p.ndim< 2 and p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{"params":decay,"weight_decay":self.hparams.weight_decay},
             {"params":nodecay,"weight_decay":0.0}],
            lr=self.hparams.learning_rate, betas=(0.9,0.95))
        def lr_lambda(step):
            warmup = int(self.hparams.warmup_ratio * max(self.trainer.estimated_stepping_batches,1))
            if step < warmup: return step/max(1,warmup)
            p = (step-warmup)/max(1,self.trainer.estimated_stepping_batches-warmup)
            return 0.1+0.9*0.5*(1+math.cos(math.pi*p))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval":"step"}}
