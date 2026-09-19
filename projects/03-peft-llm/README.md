# Parameter-Efficient Fine-Tuning of Large Language Models

> One factory, five PEFT methods, and a measurement harness that reports exactly
> how many parameters each one actually trains.

**Published in:** *International Journal of Artificial Intelligence and Neural Networks (IJAINN)*,
August 2025 — *Methods, Challenges, and Future Directions*

---

## What this is

Full fine-tuning of a multi-billion-parameter model is out of reach for most
people with most budgets. The PEFT literature answers this with a family of
methods that freeze the base model and train a small number of new or selected
parameters instead — but the methods make different trade-offs, and papers
rarely put them side by side under identical conditions.

This repository implements five of them behind one interface, loads the base
model in 4-bit NF4, and instruments the result so the efficiency claims are
measured rather than asserted.

| Method | Idea | Trained parameters |
|---|---|---|
| **LoRA** | Low-rank update `ΔW = BA` on attention projections | `r·(d_in + d_out)` per target module |
| **Prefix Tuning** | Prepend learned key/value vectors to every layer | `num_virtual_tokens × layers × 2 × d` |
| **IA³** | Learned per-channel rescaling of K, V, and FFN activations | One vector per target module |
| **AdaLoRA** | LoRA with an SVD-based budget that reallocates rank during training | Adaptive |
| **Prompt Tuning** | Learned soft prompt at the input only | `num_virtual_tokens × d` |

## Design

`src/models/peft_factory.py` is the centre of the project:

```python
class PEFTFactory:
    SUPPORTED = {"lora", "prefix_tuning", "ia3", "adalora", "prompt_tuning"}
```

`load_base_model` brings the backbone up under a `BitsAndBytesConfig` with NF4
quantisation and double quantisation enabled — this is what makes the whole
thing fit on a single consumer GPU. `PEFTFactory` then wraps it with whichever
adapter the config names, so switching methods is a one-line change:

```bash
python -m src.training.train model.peft_method=ia3
python -m src.training.train model.peft_method=lora lora.r=32 lora.lora_alpha=64
```

`PEFTParameterAnalyzer.compute_efficiency_metrics` reports `total`,
`trainable`, `frozen`, `trainable_ratio`, and `compression_ratio` for whatever
is currently loaded. Those numbers go to MLflow alongside the loss curves, so
every run carries its own efficiency receipt.

## Data

`src/data/instruction_dataset.py` formats examples with the Alpaca template and
— importantly — masks the instruction span out of the loss:

```python
labels[:instr_len] = -100
```

Without that line the model spends most of its gradient budget learning to
reproduce the prompt, which is not the task. It is a small detail that makes a
large difference in instruction-tuning quality.

## Training

This project uses the HuggingFace `Trainer` rather than Lightning — PEFT, TRL,
and the `transformers` ecosystem are built around it, and fighting that
integration would have bought nothing. Cosine LR schedule, 3% warmup, batch
size 8 with 4-step accumulation, 2e-4, max grad norm 1.0, evaluation every 500
steps, early stopping with patience 3.

## Key files

| Path | What's in it |
|---|---|
| `src/models/peft_factory.py` | `PEFTFactory`, `PEFTParameterAnalyzer`, 4-bit loading |
| `src/data/instruction_dataset.py` | `InstructionDataset`, Alpaca templating, label masking |
| `src/training/train.py` | HF `Trainer` loop, MLflow integration |
| `src/inference/peft_predictor.py` | Adapter loading + generation |
| `src/api/app.py` | FastAPI service |

## Quickstart

```bash
pip install -r requirements.txt

python -m src.training.train                        # LoRA by default
python -m src.training.train model.peft_method=adalora

pytest

uvicorn src.api.app:app --port 8002
docker compose up
```

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/instruct` | Instruction → completion |
| `POST` | `/instruct/batch` | Batched |

## Configuration

`src/config/config.yaml`:

| Key | Default |
|---|---|
| `model.base_model` | `microsoft/phi-2` |
| `model.peft_method` | `lora` |
| `lora.r` / `lora.lora_alpha` | 16 / 32 |
| `lora.target_modules` | `[q_proj, v_proj, k_proj, o_proj]` |
| `prefix_tuning.num_virtual_tokens` | 20 |
| `ia3.target_modules` | `[k_proj, v_proj, down_proj]` |
| `data.max_seq_length` | 512 |
| `training.learning_rate` | 2e-4 |

## Stack

PyTorch 2.3 · `transformers` 4.41 · `peft` 0.11 · `bitsandbytes` · `trl` ·
`accelerate` · `datasets` · `evaluate` + `rouge-score` · Hydra · MLflow ·
FastAPI · pytest · Docker
