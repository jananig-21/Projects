"""Lightning training module for WELC Classifier."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAUROC

from src.models.welc_layer import WELCClassifier


class WELCLightningModule(L.LightningModule):
    """
    Training module for Weight-Embedded Logic Circuit classifier.
    Includes auxiliary loss to regularize weight magnitudes toward ±1
    (encourages clean binarization).
    """

    def __init__(
        self,
        num_classes: int = 10,
        weight_mode: str = "binary",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        binarization_loss_weight: float = 0.1,
        warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = WELCClassifier(
            num_classes=num_classes,
            weight_mode=weight_mode,
        )

        self.binarization_loss_weight = binarization_loss_weight

        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1    = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_prec  = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_rec   = MulticlassRecall(num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _binarization_regularizer(self) -> torch.Tensor:
        """
        Regularization: push latent weights toward ±1 or 0.
        Loss = E[( |w| - 1 )^2 ] (penalizes weights far from ±1).
        This encourages clean binary representations.
        """
        reg = torch.tensor(0.0, device=self.device)
        for name, module in self.model.named_modules():
            if hasattr(module, "weight_latent"):
                w = module.weight_latent
                reg = reg + ((w.abs() - 1.0) ** 2).mean()
        return reg

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        ce_loss = F.cross_entropy(logits, y)
        bin_loss = self._binarization_regularizer()
        loss = ce_loss + self.binarization_loss_weight * bin_loss

        preds = logits.argmax(dim=-1)
        self.train_acc(preds, y)

        self.log("train/loss",       loss,     prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/ce_loss",    ce_loss,  on_step=False, on_epoch=True)
        self.log("train/bin_loss",   bin_loss, on_step=False, on_epoch=True)
        self.log("train/acc",        self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)

        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_prec(preds, y)
        self.val_rec(preds, y)

        self.log("val/loss", loss,          prog_bar=True, on_epoch=True)
        self.log("val/acc",  self.val_acc,  prog_bar=True, on_epoch=True)
        self.log("val/f1",   self.val_f1,   prog_bar=True, on_epoch=True)
        self.log("val/prec", self.val_prec, on_epoch=True)
        self.log("val/rec",  self.val_rec,  on_epoch=True)

    def configure_optimizers(self) -> Any:
        # Separate lr for latent weights vs scaling factors
        latent_params  = [p for n, p in self.named_parameters() if "weight_latent" in n]
        scaling_params = [p for n, p in self.named_parameters() if "scale" in n]
        other_params   = [p for n, p in self.named_parameters()
                          if "weight_latent" not in n and "scale" not in n]

        optimizer = torch.optim.AdamW([
            {"params": latent_params,  "lr": self.hparams.learning_rate,       "weight_decay": 0.0},
            {"params": scaling_params, "lr": self.hparams.learning_rate * 10,  "weight_decay": 0.0},
            {"params": other_params,   "lr": self.hparams.learning_rate,       "weight_decay": self.hparams.weight_decay},
        ])

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.hparams.learning_rate, self.hparams.learning_rate * 10, self.hparams.learning_rate],
            epochs=self.hparams.max_epochs,
            steps_per_epoch=self.trainer.estimated_stepping_batches // self.hparams.max_epochs or 1,
            pct_start=self.hparams.warmup_epochs / self.hparams.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
