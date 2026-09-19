"""Lightning module for Process-Aware Transformer fault classifier."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassRecall, MulticlassAUROC,
)

from src.models.process_aware_transformer import ProcessAwareTransformer


class PATLightningModule(L.LightningModule):
    def __init__(
        self,
        num_sensors: int, window_size: int, process_feature_dim: int, num_classes: int,
        d_model: int = 128, nhead: int = 8, num_encoder_layers: int = 4,
        dim_feedforward: int = 512, dropout: float = 0.1, process_embed_dim: int = 64,
        patch_size: int = 5, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
        warmup_steps: int = 200, max_epochs: int = 100, focal_gamma: float = 2.0,
        localization_loss_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = ProcessAwareTransformer(
            num_sensors=num_sensors, window_size=window_size,
            process_feature_dim=process_feature_dim, num_classes=num_classes,
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            process_embed_dim=process_embed_dim, patch_size=patch_size,
        )

        self.focal_gamma              = focal_gamma
        self.localization_loss_weight = localization_loss_weight

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1    = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_prec  = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_rec   = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_auroc = MulticlassAUROC(num_classes=num_classes)

    def forward(self, sensor_window, process_features):
        return self.model(sensor_window, process_features)

    def _focal_loss(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        return ((1 - pt) ** self.focal_gamma * ce).mean()

    def _step(self, batch, prefix):
        out    = self(batch["sensor_window"], batch["process_features"])
        labels = batch["label"]

        cls_loss = self._focal_loss(out["fault_logits"], labels)
        s_var    = batch["sensor_window"].var(dim=1)
        s_norm   = s_var / (s_var.max(dim=1, keepdim=True).values + 1e-8)
        loc_loss = F.mse_loss(out["sensor_anomaly_scores"], s_norm)
        total    = cls_loss + self.localization_loss_weight * loc_loss

        preds = out["fault_logits"].argmax(dim=-1)
        probs = out["fault_logits"].softmax(dim=-1)

        if prefix == "train":
            self.train_acc(preds, labels)
            self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.val_acc(preds, labels);   self.val_f1(preds, labels)
            self.val_prec(preds, labels);  self.val_rec(preds, labels)
            self.val_auroc(probs, labels)
            self.log("val/acc",   self.val_acc,   prog_bar=True, on_epoch=True)
            self.log("val/f1",    self.val_f1,    prog_bar=True, on_epoch=True)
            self.log("val/prec",  self.val_prec,  on_epoch=True)
            self.log("val/rec",   self.val_rec,   on_epoch=True)
            self.log("val/auroc", self.val_auroc, on_epoch=True)

        self.log(f"{prefix}/loss",     total,    prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}/cls_loss", cls_loss, on_step=False, on_epoch=True)
        self.log(f"{prefix}/loc_loss", loc_loss, on_step=False, on_epoch=True)
        return total

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
            lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        def lr_fn(step):
            w = self.hparams.warmup_steps
            t = self.trainer.estimated_stepping_batches
            if step < w:
                return step / max(1, w)
            p = (step - w) / max(1, t - w)
            return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
