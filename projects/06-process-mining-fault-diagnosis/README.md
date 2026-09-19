# Process Mining Driven Modelling & Simulation for Fault Diagnosis in Cyber-Physical Systems

> Treat a CPS sensor stream as a **process**, mine its control-flow model, and
> fuse that discrete structure with a Transformer over the raw telemetry.

**Journal:** *IoT and Cyber-Physical Systems* — accepted, indexing in progress

---

## The gap this closes

There are two well-developed ways to look at an industrial cyber-physical
system, and they rarely talk to each other.

**Signal processing / deep learning** sees a multivariate time series. It is
excellent at picking up the spectral fingerprint of a bearing starting to fail,
and it is a black box about *why*.

**Process mining** sees a sequence of discrete events and recovers a model of
the system's control flow — a Petri net, a directly-follows graph — with
conformance measures that tell you where reality diverged from the model. It is
interpretable, and it throws away the analogue detail where most fault signatures
actually live.

This project builds a bridge. Sensor telemetry is discretised into a
pm4py-compatible event log, a process model is mined from it, and
process-structural features are fused with a Transformer encoder over the raw
signal through cross-modal attention. The result keeps the deep model's
sensitivity and gains a diagnosis you can trace back to a control-flow
deviation.

```
        raw multi-sensor telemetry (W × C)
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
  TemporalEncoder          CPSEventLogGenerator
  (pre-norm Transformer,   threshold crossings → activities
   CLS token)              → pm4py event log
        │                        │
        │                        ▼
        │                 ProcessModelDiscovery
        │                 (α / heuristics / inductive)
        │                 + token-replay conformance
        │                 + DirectlyFollowsGraphAnalyzer
        │                        │
        │                        ▼
        │                 ProcessFeatureBranch
        │                        │
        └────────┬───────────────┘
                 ▼
         CrossModalFusion   (learnable sigmoid gate)
                 ▼
          fault classification
```

## Turning signals into an event log

`src/data/cps_simulator.py` is the largest single file in the project and does
the conceptual heavy lifting.

`CPSFaultSimulator` generates eight conditions — `normal`, `sensor_drift`,
`actuator_stuck`, `communication_delay`, `oscillation`, `overload`,
`partial_failure`, `cascade_failure` — each with its own injection method rather
than a generic noise knob, so the fault classes are genuinely distinguishable by
mechanism and not just by amplitude.

`CPSEventLogGenerator` then converts telemetry into discrete activities at
threshold crossings and emits a DataFrame with the standard pm4py column names:

```
case:concept:name   concept:name   time:timestamp
```

which means the log can be handed to any process-mining tool unchanged, not just
to this codebase. `extract_process_features` produces a fixed 30-dimensional
vector — **20 activity frequencies + 6 inter-arrival statistics + 4 complexity
measures** — which is what the model's process branch consumes.

## Process mining

`src/models/process_miner.py`:

- **`ProcessModelDiscovery`** — α-algorithm, heuristics miner, and inductive
  miner, with token-replay conformance checking against the discovered net.
- **`DirectlyFollowsGraphAnalyzer`** — DFG construction and analysis.
- A `PM4PY_AVAILABLE` flag with a **graceful degradation path**, so the module
  imports and the rest of the pipeline runs even where pm4py is not installed.
  (Note: pm4py is not pinned in `requirements.txt` — install it explicitly to
  use the mining features.)

## Models

Two architectures live here, and they are genuinely different designs rather
than versions of one thing.

**`ProcessFaultTransformer`** (`src/models/process_fault_transformer.py`) —
late fusion. A `TemporalEncoder` (pre-norm Transformer with a CLS token and
sinusoidal positional encoding) embeds the telemetry; a `ProcessFeatureBranch`
embeds the 30-d process vector; `CrossModalFusion` combines them through a
learnable sigmoid gate that decides, per sample, how much to trust the process
signal. Returns `{logits, embedding}`.

**`ProcessAwareTransformer`** (`src/models/process_aware_transformer.py`) —
early fusion. `SensorPatchEmbedding` splits the window into patches, ViT-style;
`ProcessContextCrossAttention` lets the patch tokens attend to the process
context at every layer instead of only at the head. Dual output heads:
`fault_logits` **and** `sensor_anomaly_scores`, so the model says both *what*
failed and *which sensor* shows it.

Trained by `PATLightningModule` with **focal loss (γ = 2.0)** — fault classes
are heavily imbalanced by construction and plain cross-entropy lets the majority
class dominate — plus a localisation MSE term at weight 0.3 for the
sensor-anomaly head.

## Known inconsistency

The source archive contains two parallel model lineages, and they are not fully
wired together. This is preserved as-is rather than silently "fixed":

- `src/training/train.py` imports `ProcessFaultLightningModule` from
  `src.training.lightning_module`, but that module defines
  **`PATLightningModule`** (wrapping `ProcessAwareTransformer`). The training
  entry point will raise `ImportError` until one name is aligned to the other.
- `src/inference/fault_predictor.py` and `tests/test_process_mining.py` use
  `ProcessFaultTransformer`, the other lineage.
- Similarly there are two dataset modules: `src/data/fault_dataset.py`
  (`CPSConfig`-based, used by `train.py` and the tests) and
  `src/data/cps_dataset.py` (`CPSSimulatorConfig`-based, a 5-class variant:
  `normal`, `bearing_fault`, `gear_fault`, `rotor_imbalance`, `sensor_drift`).

The inference path and the test suite are internally consistent; the training
entry point needs the one-line rename to match whichever lineage you want to
train.

## Key files

| Path | What's in it |
|---|---|
| `src/data/cps_simulator.py` | `CPSFaultSimulator`, `CPSEventLogGenerator`, 8 fault types, 30-d process features |
| `src/data/fault_dataset.py` | `CPSFaultDataset`, `create_fault_dataloaders`, class weights |
| `src/data/cps_dataset.py` | 5-class variant: `CPSSignalSimulator`, `ProcessEventLogExtractor` |
| `src/models/process_miner.py` | `ProcessModelDiscovery`, `DirectlyFollowsGraphAnalyzer` |
| `src/models/process_fault_transformer.py` | `TemporalEncoder`, `ProcessFeatureBranch`, `CrossModalFusion` |
| `src/models/process_aware_transformer.py` | `SensorPatchEmbedding`, `ProcessContextCrossAttention`, dual heads |
| `src/training/lightning_module.py` | `PATLightningModule`, focal loss + localisation MSE |
| `src/api/app.py`, `src/api/schemas.py` | FastAPI service |
| `tests/test_process_mining.py` | Simulator, event log, dataset, and model tests |

## Quickstart

```bash
pip install -r requirements.txt
pip install pm4py          # not pinned in requirements.txt; needed for mining

pytest                     # simulator, event-log, dataset and model tests

uvicorn src.api.app:app --port 8005
docker compose up

# Training: see "Known inconsistency" above first
python -m src.training.train
```

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/diagnose` | Telemetry window → fault class + confidence |
| `POST` | `/diagnose/batch` | Batched |
| `GET` | `/fault-types` | The supported fault taxonomy |

## Configuration

`src/config/config.yaml`:

| Key | Default |
|---|---|
| `cps.num_sensors` / `cps.window_size` | 20 / 50 |
| `cps.sampling_rate_hz` | 100 |
| `process_mining.discovery_algorithm` | `alpha` (\| `heuristics` \| `inductive`) |
| `process_mining.conformance_method` | `token_replay` |
| `model.d_model` / `nhead` / `num_encoder_layers` | 128 / 8 / 4 |
| `model.process_embedding_dim` | 64 |
| `data.imbalance_strategy` | `oversample` (\| `class_weights` \| `focal_loss`) |

## Stack

PyTorch 2.3 · Lightning 2.2 · torchmetrics · pm4py · pandas · SciPy ·
scikit-learn · Hydra · MLflow · FastAPI · pytest · Docker
