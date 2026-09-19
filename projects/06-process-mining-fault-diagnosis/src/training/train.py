"""Training pipeline for Process Mining Fault Diagnosis."""

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
from src.data.cps_simulator import CPSConfig
from src.data.fault_dataset import create_fault_dataloaders
from src.training.lightning_module import ProcessFaultLightningModule
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    configure_logging(settings.log_level)
    set_seed(cfg.project.seed)
    log.info("Starting Process Mining Fault Diagnosis training")

    cps_cfg = CPSConfig(
        num_sensors=cfg.cps.num_sensors,
        num_actuators=cfg.cps.num_actuators,
        sampling_rate_hz=cfg.cps.sampling_rate_hz,
        window_size=cfg.model.window_size,
        seed=cfg.project.seed,
    )

    train_loader, val_loader, test_loader, dataset = create_fault_dataloaders(
        num_samples=cfg.data.num_samples,
        config=cps_cfg,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        use_process_features=cfg.model.use_process_features,
    )

    class_weights = dataset.get_class_weights()
    log.info(
        "Dataset ready",
        num_samples=len(dataset),
        num_channels=dataset.num_channels,
        process_feature_dim=dataset.process_feature_dim,
    )

    model = ProcessFaultLightningModule(
        num_channels=dataset.num_channels,
        num_fault_classes=cfg.model.num_fault_classes,
        process_feature_dim=dataset.process_feature_dim,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_encoder_layers=cfg.model.num_encoder_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        window_size=cfg.model.window_size,
        use_process_features=cfg.model.use_process_features,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs,
        warmup_epochs=cfg.training.warmup_epochs,
        label_smoothing=cfg.training.label_smoothing,
        class_weights=class_weights,
    )

    artifact_dir = Path(cfg.project.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(artifact_dir / "checkpoints"),
            filename="pfm-{epoch:03d}-{val/f1:.4f}",
            monitor="val/f1",
            mode="max",
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val/f1",
            patience=cfg.training.early_stopping_patience,
            mode="max",
        ),
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
        log_every_n_steps=20,
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)

    # Test evaluation
    test_results = trainer.test(model, test_loader, ckpt_path="best")
    log.info("Test results", results=test_results)

    # Save full checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "hyperparameters": dict(model.hparams),
        "num_channels": dataset.num_channels,
        "process_feature_dim": dataset.process_feature_dim,
        "tel_mean": dataset.tel_mean.tolist(),
        "tel_std": dataset.tel_std.tolist(),
        "pm_mean": dataset.pm_mean.tolist() if dataset.process_feat is not None else None,
        "pm_std": dataset.pm_std.tolist() if dataset.process_feat is not None else None,
    }, artifact_dir / "best_model.pt")

    log.info("Training complete")


if __name__ == "__main__":
    train()
