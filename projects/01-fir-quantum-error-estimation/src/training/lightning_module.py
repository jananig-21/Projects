"""Lightning training module for FIR Quantum Error Estimator."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from src.models.fir_estimator import FIRQuantumErrorEstimator

class FIRQuantumLightningModule(L.LightningModule):
    def __init__(self, input_dim, num_states, hidden_dims=None, filter_order=32,
                 learning_rate=1e-3, weight_decay=1e-4, warmup_steps=100,
                 max_epochs=100, lambda_dist=1.0, lambda_mag=0.5, lambda_cls=0.3):
        super().__init__()
        self.save_hyperparameters()
        self.model = FIRQuantumErrorEstimator(
            input_dim=input_dim, num_states=num_states,
            hidden_dims=hidden_dims or [256, 128, 64], filter_order=filter_order)
        self.lambda_dist = lambda_dist
        self.lambda_mag = lambda_mag
        self.lambda_cls = lambda_cls
        self.train_mae = MeanAbsoluteError()
        self.val_mae   = MeanAbsoluteError()
        self.val_f1    = MulticlassF1Score(num_classes=4, average="macro")

    def forward(self, x): return self.model(x)

    def _compute_loss(self, outputs, batch):
        targets = batch["targets"]
        noise_types = batch["noise_type"]
        num_states = outputs["error_distribution"].shape[-1]
        ideal_dist = targets[:, :num_states].clamp(1e-8, 1.0)
        error_mag  = targets[:, num_states]
        pred_dist  = outputs["error_distribution"].clamp(1e-8, 1.0)
        dist_loss = F.kl_div(pred_dist.log(), ideal_dist, reduction="batchmean")
        mag_loss  = F.huber_loss(outputs["error_magnitude"], error_mag)
        cls_loss  = F.cross_entropy(outputs["noise_class_logits"], noise_types)
        total = self.lambda_dist*dist_loss + self.lambda_mag*mag_loss + self.lambda_cls*cls_loss
        return total, {"dist_loss": dist_loss, "mag_loss": mag_loss, "cls_loss": cls_loss}

    def training_step(self, batch, batch_idx):
        outputs = self(batch["features"])
        loss, ld = self._compute_loss(outputs, batch)
        self.train_mae(outputs["error_magnitude"], batch["targets"][:, -1])
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train/mae",  self.train_mae, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["features"])
        loss, _ = self._compute_loss(outputs, batch)
        self.val_mae(outputs["error_magnitude"], batch["targets"][:, -1])
        self.val_f1(outputs["noise_class_logits"], batch["noise_type"])
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/mae",  self.val_mae, prog_bar=True, on_epoch=True)
        self.log("val/f1",   self.val_f1,  prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
            lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
