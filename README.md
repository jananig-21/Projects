<div align="center">

# Machine Learning Research — Implementations

**Six peer-reviewed papers, six production-grade implementations.**

Every project below is a paper I authored, implemented end-to-end: model code,
training pipeline, tests, a serving API, and a container image.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.2-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## The projects

| # | Project | Contribution | Venue |
|:-:|---|---|---|
| **01** | [**FIR Filters for Quantum Error Estimation**](projects/01-fir-quantum-error-estimation) | Learnable FIR filter bank — initialised from a classical `firwin` design, then trained — that estimates error syndrome, magnitude, and noise channel from quantum measurement traces | *IJET*, Oct 2025 |
| **02** | [**Recurrent Linear Attention Cells**](projects/02-recurrent-linear-attention) | Attention in **O(N) time, O(1) memory per step** via a fixed-size recurrent state, replacing the O(N²) softmax and its growing KV cache | *IJCSE* vol. 185, Sep 2025 |
| **03** | [**Parameter-Efficient Fine-Tuning of LLMs**](projects/03-peft-llm) | Five PEFT methods (LoRA, Prefix, IA³, AdaLoRA, Prompt) behind one factory, on a 4-bit NF4 backbone, with measured efficiency metrics per run | *IJAINN*, Aug 2025 |
| **04** | [**Weight Embedded Logic Circuits**](projects/04-weight-embedded-logic-circuits) | Binary/ternary networks trained with straight-through estimators and a binarisation regulariser, **exportable as logic circuits** — XNOR + popcount instead of MAC | *IJET* vol. 8(2), May 2025, pp. 112–128 |
| **05** | [**Kolmogorov–Arnold Networks**](projects/05-kolmogorov-arnold-networks) | B-spline edge functions via Cox–de Boor recursion, percentile-adaptive knots, and a symbolic-extraction path for interpretability | *Springer AI Review* — under review |
| **06** | [**Process Mining for CPS Fault Diagnosis**](projects/06-process-mining-fault-diagnosis) | Sensor telemetry → pm4py event log → mined process model, fused with a Transformer through cross-modal attention | *IoT and Cyber-Physical Systems* — accepted |

## What these have in common

Each project is built to the same structure, because the point was never just to
get a number on a benchmark — it was to make the research **reproducible and
runnable by someone else**.

```
projects/<name>/
├── src/
│   ├── models/          # the architecture — the actual contribution
│   ├── data/            # datasets, simulators, feature engineering
│   ├── training/        # Lightning module + Hydra entry point
│   ├── inference/       # checkpoint loading, batched prediction
│   ├── api/             # FastAPI service, pydantic v2 schemas
│   ├── config/          # config.yaml + pydantic-settings
│   └── utils/           # seeding, structured logging
├── tests/               # pytest — shapes, gradients, finiteness, API
├── Dockerfile           # multi-stage build
├── docker-compose.yml   # service + MLflow + Postgres + Redis
├── pyproject.toml
└── requirements.txt     # fully pinned
```

The conventions that run through all six:

- **Hydra + OmegaConf** for configuration — every hyperparameter is overridable
  from the command line, nothing is hardcoded in a training script.
- **Seeded everywhere.** `src/utils/seed.py` is byte-identical across all six
  projects; `seed: 42` is in every config.
- **MLflow** tracking on every run, with the experiment name in config.
- **structlog** with a JSON renderer in production and a console renderer in
  development — logs that a log aggregator can actually parse.
- **Pinned dependencies.** Exact versions, not ranges.
- **Tests that check the things that actually break**: output shapes, gradient
  finiteness, NaN-freedom, and API contracts — not just "does it import".

## Running any of them

Every project follows the same three commands:

```bash
cd projects/<project-name>
pip install -r requirements.txt

python -m src.training.train          # train (Hydra: override anything)
pytest                                # test
uvicorn src.api.app:app --port <port> # serve
```

or, for the full stack including the MLflow tracking server:

```bash
docker compose up
```

Ports are allocated so all six can run side by side:

| Project | Port |
|---|:-:|
| 01 · FIR Quantum Error Estimation | 8000 |
| 02 · Recurrent Linear Attention | 8001 |
| 03 · PEFT LLM | 8002 |
| 04 · Weight Embedded Logic Circuits | 8003 |
| 05 · Kolmogorov–Arnold Networks | 8004 |
| 06 · Process Mining Fault Diagnosis | 8005 |

## Technology

| Layer | Tools |
|---|---|
| **Modelling** | PyTorch 2.3, PyTorch Lightning 2.2, torchmetrics, einops |
| **Domain** | Qiskit + Aer (quantum), HuggingFace `transformers` / `peft` / `bitsandbytes` / `trl` (LLM), pm4py (process mining), SciPy + SymPy (DSP, splines, symbolic) |
| **Experiments** | Hydra, OmegaConf, MLflow |
| **Serving** | FastAPI, uvicorn, pydantic v2, pydantic-settings |
| **Ops** | Docker multi-stage builds, docker-compose, Postgres, Redis, structlog |
| **Quality** | pytest, pytest-asyncio, httpx |

## A note on completeness

Two projects carry gaps that are documented rather than papered over, because a
portfolio that quietly invents missing code is worth less than one you can trust:

- **Project 04** is missing `src/models/welc_layer.py` and `requirements.txt`
  from the source archive. Three files import the former, so training and tests
  will not run until it is restored. The expected interface is written out in
  [that project's README](projects/04-weight-embedded-logic-circuits#known-gaps).
- **Project 06** contains two parallel model lineages, and its training entry
  point imports a class name the training module does not define. The inference
  path and test suite are internally consistent; the details are in
  [its README](projects/06-process-mining-fault-diagnosis#known-inconsistency).

## Author

**Janani Giridharan** — University of Michigan

## License

[MIT](LICENSE)
