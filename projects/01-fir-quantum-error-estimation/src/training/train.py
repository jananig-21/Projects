"""Main training script for FIR Quantum Error Estimator."""
from __future__ import annotations
from pathlib import Path
import hydra, mlflow, torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
from src.config.settings import settings
from src.data.quantum_dataset import QuantumCircuitConfig, QuantumErrorDataset, create_dataloaders
from src.training.lightning_module import FIRQuantumLightningModule
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    configure_logging(settings.log_level)
    set_seed(cfg.project.seed)
    qc_config = QuantumCircuitConfig(
        num_qubits=cfg.quantum.num_qubits, num_shots=cfg.quantum.num_shots,
        noise_prob=cfg.quantum.noise_prob, fir_order=cfg.fir.order)
    dataset = QuantumErrorDataset(num_samples=cfg.data.num_samples, config=qc_config)
    train_l, val_l, test_l = create_dataloaders(
        dataset, val_split=cfg.data.val_split, test_split=cfg.data.test_split,
        batch_size=cfg.training.batch_size, num_workers=cfg.data.num_workers)
    num_states = 2 ** cfg.quantum.num_qubits
    model = FIRQuantumLightningModule(
        input_dim=dataset.feature_dim, num_states=num_states,
        hidden_dims=list(cfg.model.hidden_dims), filter_order=cfg.fir.order,
        learning_rate=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs)
    artifact_dir = Path(cfg.project.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=str(artifact_dir/"checkpoints"),
            filename="fir-{epoch:03d}-{val/loss:.4f}", monitor="val/loss", mode="min", save_top_k=3),
        EarlyStopping(monitor="val/loss", patience=cfg.training.early_stopping_patience, mode="min"),
        LearningRateMonitor(), RichProgressBar()]
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs, callbacks=callbacks,
        logger=MLFlowLogger(experiment_name=cfg.mlflow.experiment_name,
                            tracking_uri=settings.mlflow_tracking_uri),
        precision=cfg.training.precision, gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=10, deterministic=True)
    trainer.fit(model, train_l, val_l)
    trainer.test(model, test_l, ckpt_path="best")
    torch.save({"state_dict": model.state_dict(), "hyperparameters": dict(model.hparams),
                "feature_dim": dataset.feature_dim, "num_states": num_states},
               artifact_dir / "best_model.pt")
    log.info("Training complete")

if __name__ == "__main__":
    train()
