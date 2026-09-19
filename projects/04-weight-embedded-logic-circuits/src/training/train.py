"""Training script for Weight-Embedded Logic Circuit Networks."""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from src.config.settings import settings
from src.data.cifar_dataset import get_cifar10_loaders
from src.training.lightning_module import WELCLightningModule
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    configure_logging(settings.log_level)
    set_seed(cfg.project.seed)
    log.info("Starting WELC training", weight_mode=cfg.model.weight_mode)

    train_loader, val_loader = get_cifar10_loaders(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
    )

    model = WELCLightningModule(
        num_classes=cfg.model.num_classes,
        weight_mode=cfg.model.weight_mode,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs,
        binarization_loss_weight=cfg.model.binarization_loss_weight,
        warmup_epochs=cfg.model.warmup_epochs,
    )

    artifact_dir = Path(cfg.project.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(artifact_dir / "checkpoints"),
            filename="welc-{epoch:03d}-{val/acc:.4f}",
            monitor="val/acc",
            mode="max",
            save_top_k=3,
        ),
        EarlyStopping(monitor="val/acc", patience=cfg.training.early_stopping_patience, mode="max"),
        LearningRateMonitor(logging_interval="step"),
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
        log_every_n_steps=20,
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)

    # Compute and log efficiency stats
    stats = model.model.compute_efficiency_stats()
    log.info("Hardware efficiency stats", **{k: str(v) for k, v in stats.items()})
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_id=mlf_logger.run_id):
        mlflow.log_metrics(stats)

    # Save
    torch.save({
        "state_dict": model.state_dict(),
        "hyperparameters": dict(model.hparams),
        "efficiency_stats": stats,
    }, artifact_dir / "best_model.pt")
    log.info("Training complete")


if __name__ == "__main__":
    train()
