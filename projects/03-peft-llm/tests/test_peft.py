"""Tests for PEFT LLM components."""
from __future__ import annotations
import pytest
from src.models.peft_factory import PEFTFactory, PEFTParameterAnalyzer

class TestPEFTFactory:
    @pytest.mark.parametrize("method,cfg", [
        ("lora", {"r":8,"lora_alpha":16,"target_modules":["q_proj","v_proj"]}),
        ("prefix_tuning", {"num_virtual_tokens":10}),
        ("ia3", {"target_modules":["k_proj","v_proj","down_proj"],"feedforward_modules":["down_proj"]}),
        ("prompt_tuning", {"num_virtual_tokens":8}),
    ])
    def test_create_config(self, method, cfg):
        assert PEFTFactory.create_peft_config(method, cfg) is not None

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            PEFTFactory.create_peft_config("unknown", {})

class TestPEFTAnalyzer:
    def test_metrics_keys(self):
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(10,10), nn.Linear(10,5))
        for p in list(model.parameters())[:2]: p.requires_grad = False
        metrics = PEFTParameterAnalyzer.compute_efficiency_metrics(model)
        assert all(k in metrics for k in ["total_parameters","trainable_parameters","trainable_ratio"])
        assert 0 <= metrics["trainable_ratio"] <= 1
