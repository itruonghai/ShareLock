# ShareLock Benchmark Results

Zero-shot ImageNet-1k evaluation using 80-template CLIP ensemble, OpenAI curated class names,
and BICUBIC interpolation. All runs on a single NVIDIA RTX PRO 6000 Blackwell (97 GB).

---

## Evaluation Timing

| Phase | Time |
|---|---|
| Prototype encoding (Llama-3-8B, 80K texts, batch=512) | ~1:33 min |
| Prototype encoding (from cache) | ~0s |
| Image evaluation (50K images, batch=512, 98 batches) | ~0:50 min |
| **Total (cold, no cache)** | **~2:23 min** |
| **Total (proto cache hit)** | **~0:50 min** |

> Prototype caches are saved automatically as `{checkpoint}_protos_openai_80tmpl.pt`.
> `checkpoints/ShareLock-CC3M.ckpt` and `logs/cc3m_llava/version_7` already had caches.

---

## Zero-Shot ImageNet-1k Results

| Checkpoint | Projector | Top-1 | Top-5 | Notes |
|---|---|---|---|---|
| `checkpoints/ShareLock-CC3M.ckpt` | MLP (4-layer, BN) | **51.47%** | **81.29%** | Published model |
| `logs/cc3m_llava/version_1` | MLP (4-layer, BN) | 49.95% | 79.51% | Best trained run |
| `logs/cc3m_qformer/version_3` | QFormer (2L, 32Q) | 48.72% | 78.41% | |
| `logs/cc3m_llava/version_7` | MLP (4-layer, BN) | 47.41% | 75.56% | |
| `logs/cc3m_llava/version_6` | MLP (4-layer, BN) | 47.00% | 75.83% | |
| `logs/cc3m_mlp_v2/version_0` | MLPv2 (LN+GELU) | 41.75% | 73.36% | |
| `logs/cc3m_mlp_v2/version_1` | MLPv2 (LN+GELU) | 41.75% | 73.36% | Identical to v0 |
| `logs/cc3m_nv_former/version_0` | NVFormer (1L, 32Q) | 42.80% | 73.91% | |
| Paper (ShareLock-CC3M) | MLP | ~54.5% | — | ~3% gap vs best |

---

## Architecture Overview

All experiments share the same backbone:
- **Vision encoder**: DINOv2 ViT-B/14 (frozen, 768-dim CLS token)
- **Vision projector**: Identity + L2-norm (0 layers)
- **Language encoder**: Llama-3-8B (frozen, last token hidden state)
- **Language projector**: varies per experiment (see below)
- **Embedding space**: 768-dim, both modalities L2-normalized
- **Dataset**: LLaVA-filtered CC3M (595K image-caption pairs)

---

## Training Summary: QFormer (`cc3m_qformer/version_3`)

**Language projector**: BLIP-2 style QFormer — 32 learned query tokens, 2 transformer layers
(self-attention on queries + cross-attention to Llama hidden states), hidden_size=512, 8 heads,
dropout=0.1, L2-norm output.

| Hyperparameter | Value |
|---|---|
| Max steps | 5000 |
| LR schedule | Cosine, warmup 150 steps |
| Learning rate | 0.001 |
| Batch size | 32768 |
| Weight decay | 0.0001 |
| Early stopping | Yes (patience=50, min_delta=0.1) |
| Gradient clip | 1.0 |
| GPUs | 1 |
| Precision | bf16-mixed |
| Val check interval | every 500 steps |

**Result**: 48.72% top-1 — competitive with the standard MLP despite very different architecture.
The cross-attention pooling to 32 queries provides a compact but expressive text representation.
Versions 0–2 did not produce saved checkpoints (early stopping triggered before first save, or
training diverged).

---

## Training Summary: MLPv2 (`cc3m_mlp_v2/version_0` and `version_1`)

**Language projector**: MLPv2 — 4-layer MLP with LayerNorm + GELU activations + residual
connections, hidden_size=4096, dropout=0.1, L2-norm output. Replaces BatchNorm+ReLU from the
standard MLP.

| Hyperparameter | Value |
|---|---|
| Max steps | 5000 |
| LR schedule | Cosine, warmup 150 steps |
| Learning rate | 0.001 |
| Batch size | 32768 |
| Weight decay | 0.0001 |
| Early stopping | Yes (patience=50, min_delta=0.1) |
| Gradient clip | 1.0 |
| GPUs | 1 |
| Precision | bf16-mixed |

**Result**: 41.75% top-1 for both version_0 and version_1 (identical). Significantly lower than
the standard MLP (49.95% best). The LayerNorm + GELU + residual design underperforms BatchNorm +
ReLU in this setting — likely because BatchNorm's running statistics provide implicit regularization
well-suited to the frozen-encoder contrastive setup, whereas LayerNorm normalizes per-sample and
loses batch-level structure.

---

## NV-Former Training (`cc3m_nv_former`) — In Progress

**Language projector**: Single cross-attention pooling layer — 32 learned query tokens
cross-attending to Llama hidden states, hidden_size=512, 8 heads, L2-norm output.
Lighter than QFormer (no self-attention, no transformer stack).

| Hyperparameter | Value |
|---|---|
| Max steps | 5000 |
| LR schedule | Cosine, warmup 150 steps |
| Learning rate | 0.001 |
| Batch size | 32768 |
| Weight decay | 0.0001 |
| Early stopping | Yes (patience=50, min_delta=0.1) |
| GPUs | **2 (DDP)** |
| Precision | bf16-mixed |

**Result**: 42.80% top-1 / 73.91% top-5.

Training ran full 5000 steps (no early stopping). Train loss: 10.04 → 5.79. Val loss: 6.35 → 6.06.
Marginally better than MLPv2 (41.75%) but well below QFormer (48.72%) and standard MLP (49.95%).
The single cross-attention layer without the QFormer's self-attention stack appears insufficient
to fully leverage the Llama hidden states.

---

## Evaluation Commands

```bash
# Single checkpoint (uses proto cache if available)
./run_experiment.sh --eval-only <checkpoint.ckpt> --config <config.yaml>

# All checkpoints (parallel, 2 GPUs)
./run_all_evals.sh
```

Logs saved to `eval_logs/<name>.log`.
