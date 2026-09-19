"""Training script for Kolmogorov-Arnold Networks."""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from src.config.settings import settings
from src.data.function_dataset import create_kan_dataloaders, RealWorldDataset
from src.training.lightning_module import KANLightningModule
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    configure_logging(settings.log_level)
    set_seed(cfg.project.seed)
    log.info("Starting KAN training", task=cfg.data.task, dataset=cfg.data.dataset)

    artifact_dir = Path(cfg.project.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Data
    if cfg.data.dataset == "synthetic":
        train_loader, val_loader, test_loader, input_dim = create_kan_dataloaders(
            function_name="kat_2d",
            num_samples=cfg.data.num_samples,
            noise_std=cfg.data.noise_std,
            batch_size=cfg.training.batch_size,
            val_split=cfg.data.val_split,
            test_split=cfg.data.test_split,
            num_workers=cfg.data.num_workers,
        )
        output_dim = 1
    else:
        from torch.utils.data import DataLoader, random_split
        full_ds   = RealWorldDataset(name=cfg.data.dataset)
        input_dim  = full_ds.input_dim
        output_dim = 1
        total      = len(full_ds)
        test_size  = int(total * cfg.data.test_split)
        val_size   = int(total * cfg.data.val_split)
        train_size = total - val_size - test_size
        train_ds, val_ds, test_ds = random_split(
            full_ds, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        kw = {"num_workers": cfg.data.num_workers, "pin_memory": torch.cuda.is_available()}
        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,  **kw)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size, shuffle=False, **kw)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.training.batch_size, shuffle=False, **kw)

    hidden_sizes = list(cfg.model.layer_sizes)[1:-1]
    layer_sizes  = [input_dim] + hidden_sizes + [output_dim]
    log.info("Layer sizes", layer_sizes=layer_sizes)

    model = KANLightningModule(
        layer_sizes=layer_sizes,
        grid_size=cfg.model.grid_size,
        spline_order=cfg.model.spline_order,
        base_activation=cfg.model.base_activation,
        grid_range=list(cfg.model.grid_range),
        scale_noise=cfg.model.scale_noise,
        dropout=cfg.model.dropout,
        task=cfg.data.task,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs,
        T_0=cfg.training.T_0,
        T_mult=cfg.training.T_mult,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(artifact_dir / "checkpoints"),
            filename="kan-{epoch:03d}-{val/loss:.5f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(monitor="val/loss", patience=cfg.training.early_stopping_patience, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        run_name=cfg.mlflow.run_name,
        tracking_uri=settings.mlflow_tracking_uri,
    )

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=mlf_logger,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)
    test_results = trainer.test(model, test_loader, ckpt_path="best")
    log.info("Test results", results=test_results)

    torch.save(
        {
            "state_dict":      model.state_dict(),
            "hyperparameters": dict(model.hparams),
            "layer_sizes":     layer_sizes,
            "input_dim":       input_dim,
            "output_dim":      output_dim,
            "symbolic_repr":   model.model.get_symbolic_representation(),
        },
        artifact_dir / "best_model.pt",
    )
    log.info("Training complete")


if __name__ == "__main__":
    train()
