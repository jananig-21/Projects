# Kolmogorov–Arnold Networks

> Learnable activation functions **on the edges**, not the nodes — parameterised
> as B-splines, with adaptive knots and a symbolic export path.

**Status:** under review at *Springer Artificial Intelligence Review*
**Authors:** Janani Giridharan, Jayachandiran U.
*"KANs: Towards Interpretable and Efficient Function Approximation Beyond MLPs"*

---

## MLPs vs. KANs, structurally

An MLP puts a fixed nonlinearity (ReLU, GELU, …) at each **node** and learns the
linear weights on the **edges**. A Kolmogorov–Arnold Network inverts this: the
edges carry *learnable univariate functions*, and the nodes simply sum. The
justification is the Kolmogorov–Arnold representation theorem, which says any
multivariate continuous function decomposes into finite compositions and sums of
univariate ones.

The practical consequence is **interpretability**. Because every edge is a
univariate function of one input, you can plot it, and you can try to name it.
An MLP's learned representation is a tangle of high-dimensional half-spaces; a
KAN's is a set of one-dimensional curves you can actually look at.

## Implementation

The core of the project is `src/models/kan_layer.py` (~14 KB), and the core of
*that* is the B-spline basis.

**`b_spline_basis(x, grid, k)`** implements the Cox–de Boor recursion:

```
B_{i,0}(x) = 1  if  t_i ≤ x < t_{i+1},  else 0

B_{i,k}(x) = (x − t_i)/(t_{i+k} − t_i) · B_{i,k−1}(x)
           + (t_{i+k+1} − x)/(t_{i+k+1} − t_{i+1}) · B_{i+1,k−1}(x)
```

Both denominators go to zero whenever knots coincide — which happens routinely
at the boundaries of a clamped knot vector, and can happen in the interior once
the grid starts adapting. Every division in the implementation is guarded. This
is the single most common way a hand-rolled KAN silently produces NaNs.

**`KANLayer`** holds four tensors and a buffer:

| Parameter | Role |
|---|---|
| `weight_base` | Coefficient on the base activation (SiLU) path |
| `weight_spline` | B-spline coefficients — the learnable edge function |
| `scale_base`, `scale_spline` | Per-edge mixing between the two paths |
| `grid` (buffer) | Knot positions, not a gradient parameter |

`update_grid()` repositions the knots from the **percentiles of the observed
activations**, so the spline resolution follows where the data actually is
rather than sitting on a uniform grid over a range that may be mostly empty.

Also in the file: `KAN` (the stack), `KANResidualBlock` / `KANResidual` for
depth, and `get_symbolic_representation()`, which attempts to fit each learned
edge function to a named closed form — the interpretability payoff, made
concrete.

## Why `precision: 32`

The training config pins full precision, with the comment *"KAN gradients can be
unstable with fp16"*. This is not caution for its own sake. Spline coefficients
and the basis functions they multiply span a wide dynamic range, and the
Cox–de Boor recursion involves differences of nearby knot values — exactly the
pattern where fp16 catastrophically cancels. Mixed precision here buys a little
speed and costs you the run.

Regularisation is entropy + L1 on the spline coefficients, which pushes toward
*sparse* edge functions — the ones you have a chance of reading symbolically.
Schedule is `CosineAnnealingWarmRestarts(T_0=50, T_mult=2)`.

## Benchmarks

`src/data/function_dataset.py` defines `BENCHMARK_FUNCTIONS`:

| Name | What it probes |
|---|---|
| `kat_2d` | A function in Kolmogorov–Arnold form — the natural best case |
| `feynman_pendulum` | A physical law from the Feynman symbolic-regression set |
| `hartmann_2d` | Standard multi-modal optimisation benchmark |
| `composition` | Nested univariate structure |
| `poly_4d` | Higher-dimensional polynomial |
| `nguyen7` | Classic symbolic-regression target |
| `circles` | Non-monotone decision boundary |

Plus `RealWorldDataset`, wrapping scikit-learn datasets (California Housing,
Iris) so the comparison isn't purely synthetic.

## Key files

| Path | What's in it |
|---|---|
| `src/models/kan_layer.py` | `b_spline_basis`, `KANLayer`, `KAN`, `KANResidual`, symbolic export |
| `src/data/function_dataset.py` | `BENCHMARK_FUNCTIONS`, `RealWorldDataset` |
| `src/training/lightning_module.py` | Entropy + L1 regularisers, warm-restart schedule |
| `src/inference/kan_predictor.py`, `predictor.py` | Inference, feature attribution |
| `src/api/app.py`, `src/api/schemas.py` | FastAPI service, pydantic v2 schemas |

## Quickstart

```bash
pip install -r requirements.txt

python -m src.training.train
python -m src.training.train model.layer_sizes=[4,16,16,1] model.grid_size=10
python -m src.training.train data.dataset=california_housing

pytest

uvicorn src.api.app:app --port 8004
docker compose up
```

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/predict` | Single input → prediction |
| `POST` | `/predict/batch` | Batched |
| `POST` | `/feature-importance` | Per-input attribution from the edge functions |
| `GET` | `/symbolic` | Closed-form reading of the learned edges |
| `GET` | `/export/lut` | Edge functions as lookup tables (hardware / embedded targets) |

## Configuration

`src/config/config.yaml`:

| Key | Default | Note |
|---|---|---|
| `model.architecture` | `KAN` | `KAN` \| `KAN_Residual` \| `KAN_Ensemble` |
| `model.layer_sizes` | `[2, 8, 8, 1]` | |
| `model.grid_size` | 5 | B-spline grid points |
| `model.spline_order` | 3 | Cubic |
| `model.grid_range` | `[-1.0, 1.0]` | Initial knot span, then adapted |
| `training.precision` | `32` | See above — do not lower this |
| `training.lr_scheduler` | `cosine_with_restarts` | `T_0=50`, `T_mult=2` |

## Stack

PyTorch 2.3 · Lightning 2.2 · torchmetrics · SciPy · SymPy · scikit-learn ·
einops · Hydra · MLflow · FastAPI · pytest · Docker
