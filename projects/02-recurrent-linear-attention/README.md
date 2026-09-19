# Recurrent Linear Attention Cells

> Linear-complexity attention with a fixed-size recurrent state: **O(N)** time
> and **O(1)** memory per step at inference, instead of the O(N²) time and
> O(N) KV cache of softmax attention.

**Published in:** *International Journal of Computer Sciences and Engineering (IJCSE)*,
Volume 185, September 2025

---

## The problem with softmax attention on long sequences

Standard attention computes `softmax(QKᵀ/√d)V`. The `QKᵀ` matrix is N×N, so
both the compute and the memory scale quadratically in sequence length. At
inference time you also carry a KV cache that grows linearly with every token
generated. Neither is workable for genuinely long contexts.

The linear-attention trick is to drop the softmax and exploit associativity.
If you replace `exp(qᵀk)` with a feature map `φ(q)ᵀφ(k)`, then

```
    (φ(Q)φ(K)ᵀ)V   ≡   φ(Q)(φ(K)ᵀV)
     ^^^^^^^^^^^        ^^^^^^^^^^^
      N×N matrix         K×D matrix
```

and the right-hand grouping never materialises anything of size N×N. Written
recurrently, this becomes a running state that is updated one token at a time:

```
State:  S ∈ R^{K×D_v},  z ∈ R^K

    S_t = S_{t-1} + φ(k_t) ⊗ v_t
    z_t = z_{t-1} + φ(k_t)

    y_t = (φ(q_t) S_t) / (φ(q_t) z_t)
```

`S` and `z` are **fixed size**. They do not grow with t. That is the whole
point: a 2,048-token context and a 200,000-token context cost the same per-step
memory.

## Architecture

| Component | File | Detail |
|---|---|---|
| `RLAKernelFeatureMap` | `src/models/rla_cell.py` | `φ(x) = elu(Wx) + 1 + ε` — strictly positive, so the denominator can't vanish. Orthogonal init. |
| `RecurrentLinearAttentionCell` | `src/models/rla_cell.py` | The state update above, with the complexity contract documented in the docstring |
| `RLAFeedForward` | `src/models/rla_cell.py` | SwiGLU, 4× expansion |
| `RLABlock` | `src/models/rla_cell.py` | Pre-norm residual: attention → FFN |
| `RecurrentLinearAttentionModel` | `src/models/rla_cell.py` | 6 blocks, d=512, 8 heads, vocab 50257, **tied** token embedding / LM head |

The `elu(·) + 1 + ε` feature map is the load-bearing detail. It has to be
positive everywhere — the denominator `φ(q)·z` is a sum of inner products of
feature vectors, and if any component can go negative the normaliser can pass
through zero and the whole thing diverges. The `+1e-6` is not decoration.

Embedding and LM head share weights, which is both a regulariser and a
meaningful parameter saving at a 50k vocabulary.

## Training

Perplexity-monitored, with a `LambdaLR` cosine schedule and a 5% linear warmup.
Batch size 16 with 4-step gradient accumulation (effective 64), AdamW at 3e-4
with weight decay 0.1, gradient clipping at 1.0, mixed precision.

```bash
pip install -r requirements.txt

python -m src.training.train
python -m src.training.train model.num_layers=12 data.seq_len=4096

pytest

uvicorn src.api.app:app --port 8001
docker compose up
```

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/generate` | Autoregressive generation from a prompt |
| `POST` | `/generate/batch` | Batched generation |

Generation in `src/inference/generator.py` uses the recurrent formulation
directly — it carries `(S, z)` forward rather than re-attending over history,
which is where the O(1)-per-step claim actually cashes out.

## Configuration

`src/config/config.yaml`:

| Key | Default |
|---|---|
| `model.hidden_dim` | 512 |
| `model.num_layers` | 6 |
| `model.num_heads` | 8 |
| `model.max_seq_len` | 2048 |
| `model.vocab_size` | 50257 |
| `training.learning_rate` | 3e-4 |
| `training.accumulate_grad_batches` | 4 |
| `training.warmup_ratio` | 0.05 |

## Stack

PyTorch 2.3 · Lightning 2.2 · einops · HuggingFace `transformers` / `tokenizers` ·
Hydra · MLflow · FastAPI · pytest · Docker
