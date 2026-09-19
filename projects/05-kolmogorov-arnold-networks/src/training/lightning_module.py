"""Lightning training module for KAN with grid-update schedule."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from src.models.kan_layer import KAN, KANResidual


class KANLightningModule(L.LightningModule):
    """
    KAN training module with:
    - Periodic grid updates (adaptive knot placement)
    - Entropy regularization (promotes sparse activations)
    - Symbolic weight regularization (encourages interpretable functions)
    - Full regression and classification support
    """

    def __init__(
        self,
        layer_sizes: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
        grid_range: list[float] = None,
        scale_noise: float = 0.1,
        dropout: float = 0.0,
        task: str = "regression",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        T_0: int = 50,
        T_mult: int = 2,
        grid_update_every_n_epochs: int = 20,
        entropy_reg_weight: float = 1e-4,
        sparsity_reg_weight: float = 1e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        grid_range_tuple = tuple(grid_range or [-1.0, 1.0])

        self.model = KAN(
            layer_sizes=layer_sizes,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
            grid_range=grid_range_tuple,
            scale_noise=scale_noise,
            dropout=dropout,
        )

        self.task = task
        self.entropy_reg_weight  = entropy_reg_weight
        self.sparsity_reg_weight = sparsity_reg_weight

        # Metrics
        if task == "regression":
            self.train_mae = MeanAbsoluteError()
            self.val_mae   = MeanAbsoluteError()
            self.val_mse   = MeanSquaredError()
            self.val_r2    = R2Score()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _entropy_regularizer(self) -> torch.Tensor:
        """
        Encourage sparse, interpretable activations.
        Penalizes uniform activation (maximizes entropy → less interpretable).
        """
        reg = torch.tensor(0.0, device=self.device)
        for layer in self.model.kan_layers:
            # Spline coefficient entropy: prefer peaked distributions
            w = layer.weight_spline.abs()
            w_norm = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -(w_norm * (w_norm + 1e-8).log()).sum(dim=-1).mean()
            reg = reg + entropy
        return reg

    def _sparsity_regularizer(self) -> torch.Tensor:
        """L1 on spline weights → sparse edge activation → prunable network."""
        reg = torch.tensor(0.0, device=self.device)
        for layer in self.model.kan_layers:
            reg = reg + layer.weight_spline.abs().mean()
        return reg

    def _compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.task == "regression":
            return F.mse_loss(pred.squeeze(), target)
        elif self.task == "classification":
            return F.cross_entropy(pred, target.long())
        else:
            return F.mse_loss(pred.squeeze(), target)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, y = batch["x"], batch["y"]
        pred = self(x)
        task_loss = self._compute_loss(pred, y)
        ent_reg  = self._entropy_regularizer()
        spar_reg = self._sparsity_regularizer()
        loss = task_loss + self.entropy_reg_weight * ent_reg + self.sparsity_reg_weight * spar_reg

        if self.task == "regression":
            self.train_mae(pred.squeeze(), y)
            self.log("train/mae", self.train_mae, on_epoch=True, on_step=False)

        self.log("train/loss",     loss,      prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/task_loss", task_loss, on_step=False, on_epoch=True)
        self.log("train/ent_reg",  ent_reg,   on_step=False, on_epoch=True)
        self.log("train/spar_reg", spar_reg,  on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        x, y = batch["x"], batch["y"]
        pred = self(x)
        loss = self._compute_loss(pred, y)

        if self.task == "regression":
            p = pred.squeeze()
            self.val_mae(p, y)
            self.val_mse(p, y)
            self.val_r2(p, y)
            self.log("val/mae", self.val_mae, prog_bar=True, on_epoch=True)
            self.log("val/mse", self.val_mse, on_epoch=True)
            self.log("val/r2",  self.val_r2,  prog_bar=True, on_epoch=True)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        """Periodic grid update to re-align knots with data distribution."""
        interval = self.hparams.grid_update_every_n_epochs
        if self.current_epoch > 0 and self.current_epoch % interval == 0:
            # Collect sample of training data to update grid
            train_loader = self.trainer.train_dataloader
            samples = []
            for i, batch in enumerate(train_loader):
                if i >= 10:  # Use 10 batches for grid update
                    break
                samples.append(batch["x"])
            if samples:
                x_sample = torch.cat(samples).to(self.device)
                self.model.update_grids(x_sample)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.T_0,
            T_mult=self.hparams.T_mult,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
