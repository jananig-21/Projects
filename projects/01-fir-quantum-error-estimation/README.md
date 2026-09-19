# Designing Optimized FIR Filters for Quantum Error Estimation

> Learnable finite-impulse-response filter banks that read quantum measurement
> traces and estimate the error syndrome, its magnitude, and the underlying
> noise channel — in a single forward pass.

**Published in:** *International Journal of Engineering and Technology (IJET)*, October 2025

---

## Why this problem

Quantum error correction depends on knowing *what kind* of noise a device is
experiencing, not just *that* something went wrong. Syndrome extraction gives
you a bitstring; it does not tell you whether you are looking at depolarizing
noise, a bit flip, amplitude damping, or dephasing — and the distinction
changes which correction strategy is optimal.

Measurement records from a noisy device are, structurally, a **signal**. That
observation is the premise of this project: classical DSP has spent fifty years
building tools to pull structure out of noisy signals, and an FIR filter bank is
the natural instrument for separating noise processes that live in different
spectral bands.

The twist is that the filter coefficients are **not fixed**. They are
initialised from a classically designed `scipy.signal.firwin` Hamming-window
filter — a good prior — and then trained end-to-end with the rest of the
network, so the filter bank learns the spectral signature of the device it is
actually deployed on.

## Approach

```
measurement trace (T samples)
        │
        ▼
┌──────────────────────────┐
│  LearnableFIRLayer       │  conv1d, weights ← firwin(order=32, hamming)
│  (filter bank)           │  gradients flow into the taps
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  FIRFilterBank features  │  per-filter: energy, mean-|x|, variance
└──────────────────────────┘
        │
        ▼
    MLP trunk  [256 → 128 → 64], dropout 0.2
        │
   ┌────┼────────────────────┐
   ▼    ▼                    ▼
error_distribution   error_magnitude   noise_class_logits
 Softmax over 2^n     Softplus (≥0)     4-way (depolarizing /
 syndromes                              bit-flip / amplitude-
                                        damping / phase-flip)
```

Three heads, three losses, one optimisation problem:

```
L = 1.0·KLDiv(error_distribution)  +  0.5·Huber(error_magnitude)  +  0.3·CE(noise_class)
```

The KL term is the right choice for the syndrome head because the target *is* a
distribution over 2^n outcomes — cross-entropy against a hard label would throw
away the shot-noise information that 1024 shots per circuit actually give you.
Huber on the magnitude head keeps a handful of high-error outliers from
dominating the gradient. AdamW with cosine annealing, gradient clipping at 1.0,
early stopping on validation loss.

## Data

There is no public dataset of labelled quantum noise traces at this scale, so
`src/data/quantum_dataset.py` builds one. `QuantumNoiseSimulator` applies each
of the four channels to a 4-qubit register at a configurable noise probability
and samples 1024 shots; `QuantumErrorDataset` pairs the resulting traces with
their ground-truth syndrome distribution, error magnitude, and channel label.
50,000 samples, 70/15/15 split.

## Key files

| Path | What's in it |
|---|---|
| `src/models/fir_estimator.py` | `LearnableFIRLayer`, `FIRQuantumErrorEstimator` |
| `src/data/quantum_dataset.py` | `QuantumNoiseSimulator`, `FIRFilterBank`, `QuantumErrorDataset` |
| `src/training/lightning_module.py` | Composite loss, AdamW + `CosineAnnealingLR` |
| `src/training/train.py` | Hydra entry point, MLflow logging |
| `src/inference/predictor.py` | Checkpoint loading, batched inference |
| `src/api/app.py` | FastAPI service |

## Quickstart

```bash
pip install -r requirements.txt

# Train (Hydra — override anything from the CLI)
python -m src.training.train
python -m src.training.train training.max_epochs=50 fir.order=64 quantum.num_qubits=5

# Test
pytest

# Serve
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Or bring up the whole stack (API + MLflow + Postgres + Redis)
docker compose up
```

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + model-loaded check |
| `POST` | `/predict` | Single measurement trace → syndrome, magnitude, noise class |
| `POST` | `/predict/batch` | Same, vectorised over a list of traces |

## Configuration

All defaults live in `src/config/config.yaml` and are overridable on the command
line. The ones worth knowing:

| Key | Default | Note |
|---|---|---|
| `quantum.num_qubits` | 4 | Syndrome head width is 2^n |
| `quantum.num_shots` | 1024 | Shots per circuit |
| `fir.order` | 32 | Filter taps |
| `fir.window` | `hamming` | `firwin` initialisation window |
| `model.hidden_dims` | `[256, 128, 64]` | MLP trunk |
| `training.precision` | `16-mixed` | |
| `data.num_samples` | 50000 | |

## Stack

PyTorch 2.3 · PyTorch Lightning 2.2 · Qiskit + Qiskit Aer · SciPy · Hydra ·
MLflow · FastAPI · structlog · pytest · Docker
