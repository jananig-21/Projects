"""PEFT training pipeline."""
from __future__ import annotations
import os
from pathlib import Path
import hydra, mlflow, math, torch
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from src.config.settings import settings
from src.data.instruction_dataset import InstructionDataset
from src.models.peft_factory import PEFTFactory, PEFTParameterAnalyzer
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed
log = get_logger(__name__)

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    configure_logging(settings.log_level)
    set_seed(cfg.project.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    artifact_dir = Path(cfg.project.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    peft_cfg = dict(OmegaConf.to_container(cfg.get(cfg.model.peft_method, {}), resolve=True))
    peft_model, tokenizer = PEFTFactory.create_peft_model(
        cfg.model.base_model, cfg.model.peft_method, peft_cfg, use_4bit=torch.cuda.is_available())
    metrics = PEFTParameterAnalyzer.compute_efficiency_metrics(peft_model)
    log.info("Efficiency", **{k:str(v) for k,v in metrics.items()})
    train_ds = InstructionDataset(tokenizer, cfg.data.max_seq_length, use_synthetic=True)
    val_ds   = InstructionDataset(tokenizer, cfg.data.max_seq_length, use_synthetic=True)
    training_args = TrainingArguments(
        output_dir=str(artifact_dir/"hf_trainer"),
        num_train_epochs=cfg.training.max_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler,
        evaluation_strategy="steps", eval_steps=cfg.training.eval_steps,
        save_strategy="steps", save_steps=cfg.training.eval_steps,
        load_best_model_at_end=True, fp16=torch.cuda.is_available(),
        seed=cfg.project.seed, report_to="mlflow", logging_steps=50, save_total_limit=3,
        remove_unused_columns=False)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.log_params({"peft_method":cfg.model.peft_method,
                           "trainable_ratio":f"{metrics['trainable_ratio']:.4f}"})
        trainer = Trainer(model=peft_model, args=training_args,
                          train_dataset=train_ds, eval_dataset=val_ds,
                          data_collator=DataCollatorForSeq2Seq(
                              tokenizer, peft_model, pad_to_multiple_of=8,
                              return_tensors="pt", padding=True))
        trainer.train()
        peft_model.save_pretrained(str(artifact_dir/"peft_adapter"))
        tokenizer.save_pretrained(str(artifact_dir/"peft_adapter"))
        er = trainer.evaluate()
        mlflow.log_metrics({"final_eval_loss":er["eval_loss"],
                            "final_ppl": math.exp(er["eval_loss"])})
    log.info("PEFT training complete")

if __name__ == "__main__": train()
