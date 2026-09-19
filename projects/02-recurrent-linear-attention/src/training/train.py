"""Training script for RLA LM."""
from __future__ import annotations
from pathlib import Path
import hydra, torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
from src.config.settings import settings
from src.data.language_dataset import create_lm_dataloaders, LanguageModelingDataset
from src.training.lightning_module import RLALanguageModelModule
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed
log = get_logger(__name__)

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    configure_logging(settings.log_level)
    set_seed(cfg.project.seed)
    train_loader, val_loader = create_lm_dataloaders(
        seq_len=cfg.data.seq_len, batch_size=cfg.training.batch_size,
        val_split=cfg.data.val_split, num_workers=cfg.data.num_workers, use_synthetic=True)
    ds = LanguageModelingDataset(seq_len=cfg.data.seq_len, use_synthetic=True)
    model = RLALanguageModelModule(
        vocab_size=ds.vocab_size, hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers, num_heads=cfg.model.num_heads,
        max_seq_len=cfg.model.max_seq_len, dropout=cfg.model.dropout,
        learning_rate=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs)
    artifact_dir = Path(cfg.project.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=str(artifact_dir/"checkpoints"),
            filename="rla-{epoch:03d}-{val/ppl:.2f}", monitor="val/ppl", mode="min", save_top_k=3),
        EarlyStopping(monitor="val/ppl", patience=cfg.training.early_stopping_patience, mode="min"),
        LearningRateMonitor(), RichProgressBar()]
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs, callbacks=callbacks,
        logger=MLFlowLogger(experiment_name=cfg.mlflow.experiment_name,
                            tracking_uri=settings.mlflow_tracking_uri),
        precision=cfg.training.precision, gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=10, deterministic=True)
    trainer.fit(model, train_loader, val_loader)
    torch.save({"state_dict": model.state_dict(), "hyperparameters": dict(model.hparams)},
               artifact_dir/"best_model.pt")
    log.info("RLA training complete")

if __name__ == "__main__": train()
