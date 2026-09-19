"""PEFT method factory — LoRA, Prefix Tuning, IA³, AdaLoRA, Prompt Tuning."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from peft import (LoraConfig, PrefixTuningConfig, IA3Config, AdaLoraConfig,
                  PromptTuningConfig, TaskType, get_peft_model, PeftModel)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils.logger import get_logger
log = get_logger(__name__)

class PEFTFactory:
    SUPPORTED = {"lora","prefix_tuning","ia3","adalora","prompt_tuning"}

    @staticmethod
    def create_peft_config(method: str, cfg: dict):
        if method == "lora":
            return LoraConfig(r=cfg.get("r",16), lora_alpha=cfg.get("lora_alpha",32),
                lora_dropout=cfg.get("lora_dropout",0.05),
                target_modules=list(cfg.get("target_modules",["q_proj","v_proj"])),
                bias=cfg.get("bias","none"), task_type=TaskType.CAUSAL_LM)
        elif method == "prefix_tuning":
            return PrefixTuningConfig(num_virtual_tokens=cfg.get("num_virtual_tokens",20),
                                     task_type=TaskType.CAUSAL_LM)
        elif method == "ia3":
            return IA3Config(
                target_modules=list(cfg.get("target_modules",["k_proj","v_proj","down_proj"])),
                feedforward_modules=list(cfg.get("feedforward_modules",["down_proj"])),
                task_type=TaskType.CAUSAL_LM)
        elif method == "adalora":
            return AdaLoraConfig(r=cfg.get("r",12), lora_alpha=cfg.get("lora_alpha",32),
                target_r=cfg.get("target_r",8),
                target_modules=list(cfg.get("target_modules",["q_proj","v_proj"])),
                task_type=TaskType.CAUSAL_LM)
        elif method == "prompt_tuning":
            return PromptTuningConfig(num_virtual_tokens=cfg.get("num_virtual_tokens",8),
                                     task_type=TaskType.CAUSAL_LM)
        else:
            raise ValueError(f"Unknown PEFT method: {method}. Supported: {PEFTFactory.SUPPORTED}")

    @staticmethod
    def load_base_model(model_name, use_4bit=False, use_8bit=False):
        qcfg = None
        if use_4bit:
            qcfg = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4")
        elif use_8bit:
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=qcfg,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return model, tokenizer

    @classmethod
    def create_peft_model(cls, base_model_name, peft_method, peft_cfg, use_4bit=False):
        base_model, tokenizer = cls.load_base_model(base_model_name, use_4bit=use_4bit)
        peft_config = cls.create_peft_config(peft_method, peft_cfg)
        peft_model = get_peft_model(base_model, peft_config)
        trainable, total = peft_model.get_nb_trainable_parameters()
        log.info("PEFT model ready", method=peft_method,
                 trainable=f"{trainable:,}", pct=f"{100*trainable/total:.2f}%")
        return peft_model, tokenizer

    @staticmethod
    def load_peft_model(base_model_name, peft_checkpoint, device=None):
        base_model, tokenizer = PEFTFactory.load_base_model(base_model_name)
        model = PeftModel.from_pretrained(base_model, peft_checkpoint,
            device_map="auto" if torch.cuda.is_available() else None)
        model.eval()
        return model, tokenizer

class PEFTParameterAnalyzer:
    @staticmethod
    def compute_efficiency_metrics(model: nn.Module) -> dict:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable,
                "frozen_parameters": total-trainable,
                "trainable_ratio": trainable/max(total,1),
                "compression_ratio": total/max(trainable,1)}
