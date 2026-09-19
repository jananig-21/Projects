# Weight Embedded Logic Circuits for Efficient Neural Network Acceleration

> Binary and ternary neural networks whose trained weights can be exported as
> **logic circuits** — replacing multiply-accumulate with XNOR and popcount.

**Published in:** *International Journal of Engineering and Technology (IJET)*,
Volume 8, Version 2, May 2025, pp. 112–128

---

> [!IMPORTANT]
> **Two files in this project are missing from the source archive** and were not
> reconstructed, because inventing them would misrepresent what was published.
> See [Known gaps](#known-gaps) before trying to run it.

## The idea

A multiply-accumulate is expensive in silicon. If the weights of a layer are
constrained to `{-1, +1}`, the multiply disappears entirely: a product of two
±1 values is an XNOR of their sign bits, and the accumulation becomes a
population count. That is not an approximation of the arithmetic — it *is* the
arithmetic, exactly, once the constraint holds.

So the network is not merely compressed. Once trained, each layer can be
**emitted as a logic circuit** and dropped onto an FPGA or into an ASIC
datapath, with no multipliers in the inference path at all.

The catch, and the thing the project is actually about, is training. `sign(w)`
has zero gradient almost everywhere, so backpropagation through it gives you
nothing. The standard remedy is a **straight-through estimator**: forward pass
quantises, backward pass pretends the quantiser was the identity (clipped).
That keeps a latent real-valued weight around to accumulate gradient, and reads
its sign at inference.

## Training

Two things in `src/training/lightning_module.py` make the straight-through
scheme work well rather than just work.

**A binarisation regulariser.** Latent weights that hover near zero binarise
unstably — a tiny gradient flips the sign and the layer's function changes
discontinuously. So the loss pushes them away from zero:

```
L = CE(logits, y)  +  0.1 · E[(|w| − 1)²]
```

which has minima exactly at ±1 and grows as a weight drifts toward the unstable
middle. Logged separately as `train/ce_loss` and `train/bin_loss` so you can
watch the two terms trade off.

**Three parameter groups, with the scale parameters on a 10× learning rate.**
Binary layers keep a small number of real-valued scaling factors alongside the
quantised weights, and those live on a very different loss surface from the
latent weights. Giving them their own group — and a much higher LR — stops them
from being the bottleneck on convergence. `OneCycleLR` over the whole schedule.

Metrics: accuracy, macro F1, macro precision/recall, AUROC (torchmetrics),
on CIFAR-10.

## Export path

`WELCPredictor.get_logic_circuit_export()` in `src/inference/predictor.py`
serialises the trained network into its logic-circuit representation, and the
API exposes it directly:

```
GET /export/logic_circuits     → circuit description of each layer
GET /model/efficiency          → parameter count, bit-width, MAC-vs-XNOR savings
```

This is the part that turns the paper's claim into something you can hand to a
hardware engineer.

## Known gaps

The Google Drive archive this repository was assembled from is missing two
files. They are listed here rather than reconstructed:

| Missing | Consequence |
|---|---|
| `src/models/welc_layer.py` | Imported by `src/training/lightning_module.py`, `src/inference/predictor.py`, and `tests/test_welc.py`. **Training, inference, and the test suite will not run without it.** |
| `requirements.txt` | The `Dockerfile` does `COPY requirements.txt .`, so the image build fails at that step. `pyproject.toml` declares no dependency list either. |

From the call sites, `welc_layer.py` is expected to export:

```python
BinaryWeightSTE, TernaryWeightSTE, LogicCircuitOps,
WELCLinear, WELCConv2d, WELCClassifier,
binarize_weights, ternarize_weights
```

and `WELCClassifier(num_classes, weight_mode)` must expose submodules carrying a
`weight_latent` attribute, which is what the binarisation regulariser iterates
over.

Everything else in the project — the Lightning module, the CIFAR-10 data
pipeline, the predictor, the API, the tests, the Docker setup — is present and
complete.

## Key files

| Path | What's in it |
|---|---|
| `src/models/welc_layer.py` | **missing** — see above |
| `src/training/lightning_module.py` | `WELCLightningModule`, binarisation regulariser, 3-group optimiser, OneCycleLR |
| `src/data/cifar_dataset.py` | CIFAR-10 loaders and augmentation |
| `src/inference/predictor.py` | `WELCPredictor`, logic-circuit export |
| `src/api/app.py` | FastAPI service |
| `tests/test_welc.py` | STE, layer, and classifier tests |

## Quickstart

```bash
# requirements.txt is missing — install the stack manually, e.g.
pip install torch torchvision lightning torchmetrics hydra-core omegaconf \
            mlflow fastapi 'uvicorn[standard]' pydantic pydantic-settings \
            structlog rich pytest pytest-asyncio httpx

python -m src.training.train
python -m src.training.train model.weight_mode=ternary

pytest

uvicorn src.api.app:app --port 8003
```

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/predict` | Image → CIFAR-10 class |
| `GET` | `/export/logic_circuits` | Trained weights as logic-circuit description |
| `GET` | `/model/efficiency` | Compression and MAC-elimination statistics |

## Configuration

`src/config/config.yaml`:

| Key | Default |
|---|---|
| `model.weight_mode` | `binary` (`binary` \| `ternary`) |
| `model.binarization_loss_weight` | 0.1 |
| `model.warmup_epochs` | 5 |
| `training.batch_size` | 128 |
| `training.max_epochs` | 100 |
| `training.precision` | `16-mixed` |

## Stack

PyTorch 2.3 · Lightning 2.2 · torchmetrics · torchvision · Hydra · MLflow ·
FastAPI · structlog · pytest · Docker
