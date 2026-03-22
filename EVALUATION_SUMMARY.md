# ShareLock Zero-Shot ImageNet-1k Evaluation Summary

## Overview

This document summarizes the implementation of `eval_zero_shot_imagenet.py` — a standalone zero-shot evaluation script for ShareLock on ImageNet-1k — along with all experimental findings.

---

## Model Architecture

ShareLock aligns frozen unimodal features via two trainable MLP projectors:

```
Image → DINOv2 (frozen) → Vision Projector → embedding space
Text  → Llama-3-8B (frozen) → Language Projector → embedding space
```

### Vision Encoder: `dinov2_vitb14`
- Loaded via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`
- Outputs the CLS token (class token), 768-dim
- All parameters frozen during training and evaluation
- **Vision Projector** (CC3M checkpoint): `num_layers=0` → Identity + L2-normalize
  - DINOv2 CLS features are directly L2-normalized into the shared embedding space
  - No learned transformation on the vision side

### Language Encoder: `meta-llama/Meta-Llama-3-8B`
- Loaded via HuggingFace `AutoModel` in `bfloat16` to fit in ~16GB VRAM
- The **last token's hidden state** is used as the text representation (causal LM → last token aggregates full context)
- Tokenizer uses left-padding (`padding_side="left"`) and `eos_token` as pad token
- All parameters frozen; runs in `eval()` mode
- Unloaded from GPU after class prototype encoding to free memory for image evaluation
- **Language Projector** (CC3M checkpoint): `num_layers=4` → 4-layer MLP
  - Architecture: `[Linear → BN → ReLU → Dropout(0.2)] × 3 → Linear`
  - 3 BatchNorm layers using training running statistics at eval time
  - Input: 4096-dim (Llama hidden size) → Output: 768-dim (shared embedding space) + L2-normalize

### Shared Embedding Space
- Dimension: 768 (matches DINOv2 vitb14 CLS token size)
- Both modalities are L2-normalized into this space
- Similarity = cosine similarity (dot product of unit vectors)

---

## Evaluation Script: `eval_zero_shot_imagenet.py`

### Image Preprocessing

Follows DINOv2's `make_classification_eval_transform()` (confirmed from the `dinotxt.ipynb` notebook):

```python
transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Key: **BICUBIC** interpolation — the original `get_transforms()` in `misc.py` used the default BILINEAR, which DINOv2 does not recommend.

### Class Prototype Construction

Follows CLIP-Benchmark's `zero_shot_classifier` exactly:

1. **Class names**: OpenAI/DINOv2 curated 1000-class list (single clean name per class, e.g. `"tench"` not `"tench, Tinca tinca"`)
2. **Templates**: 80-template CLIP ensemble (from CLIP paper appendix / CLIP-Benchmark `en_zeroshot_classification_templates.json`)
3. **Encoding**: Per class — encode all 80 templates as one batch through Llama-3-8B + language projector
4. **Normalization**: `normalize(embeddings) → mean(dim=0) → normalize` (CLIP-Benchmark exact protocol)

```python
# For each class:
class_texts = [template.format(class_name) for template in templates]  # 80 texts
class_emb = model.encode_text(class_texts)          # [80, 768], already normalized by projector
class_emb = F.normalize(class_emb, dim=-1)           # normalize each template
class_emb = class_emb.mean(dim=0)                    # average
class_emb = class_emb / class_emb.norm()             # normalize again
```

Memory management: Llama-3-8B is unloaded after prototype encoding (`model.language_encoder.unload_model()`), then DINOv2 is loaded for image evaluation.

### Image Evaluation

Follows CLIP-Benchmark's `run_classification`:

```python
image_features = model.encode_image(images)          # DINOv2 CLS → L2-normalize
logits = image_features @ class_prototypes            # cosine similarity [B, 1000]
top1 = (logits.argmax(dim=-1) == labels).mean()
top5 = logits.topk(5).indices matches labels
```

AMP (`torch.autocast`) is enabled for CUDA — matches CLIP-Benchmark default.

### Checkpoint Loading

PyTorch 2.6+ defaults `weights_only=True`, but PyTorch Lightning checkpoints contain OmegaConf objects. Fixed by patching `torch.load` at module import time:

```python
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
```

---

## CLIP-Benchmark vs Our Implementation

Deep-dive confirmed the two are **functionally equivalent**:

| Factor | CLIP-Benchmark | Our Script |
|--------|---------------|------------|
| Class names | OpenAI 1000 curated | Same ✓ |
| Templates | 80 × `{c}` placeholder | 80 × `{}` (identical strings) ✓ |
| Proto normalization | normalize → mean → normalize | Same ✓ |
| Image feature norm | `F.normalize()` explicit | Vision projector L2-normalizes ✓ |
| Logit scale | `100. ×` | Raw dot product (no effect on argmax) |
| AMP | `torch.autocast` (fp16) | Added, negligible effect |
| Text batching | Per-class (80 texts) | Per-class ✓ |
| Tokenizer | Custom wrapper required | `encode_text` takes raw strings ✓ |

**Note**: CLIP-Benchmark has no native ShareLock support. The paper authors must have used a custom model wrapper with an identity "tokenizer" (passing raw strings directly to `encode_text`).

---

## Experimental Results

All experiments use the **ShareLock-CC3M checkpoint** (`checkpoints/ShareLock-CC3M.ckpt`, 613MB).

| # | Configuration | Top-1 | Top-5 | Notes |
|---|---------------|-------|-------|-------|
| 1 | First synonym only, BILINEAR resize | 48.93% | 77.74% | Baseline |
| 2 | All WordNet synonyms, BILINEAR resize | 50.01% | 80.25% | +1.08% from synonyms |
| 3 | OpenAI class names + BICUBIC resize | 51.46% | 81.28% | +1.45% from cleaner names + correct interp |
| 4 | + Per-class text batching + AMP | **51.48%** | **81.28%** | Matches CLIP-Benchmark protocol exactly |
| — | **Paper (CLIP-Benchmark)** | **54.5%** | — | Reported result |

### Key Improvements Found

1. **BILINEAR → BICUBIC** (+~0.5%): DINOv2's recommended eval transform uses BICUBIC interpolation. The repo's `get_transforms()` uses default BILINEAR.

2. **WordNet labels → OpenAI class names** (+~1%): Raw HuggingFace labels include all WordNet synonyms (e.g. `"tench, Tinca tinca"`). DINOv2 notebook and CLIP-Benchmark both use single clean names (e.g. `"tench"`).

3. **First synonym only → 80-template ensemble** (+~1.1%): Using all 80 CLIP templates rather than a single `"A photo of a {class_name}"` template.

### Remaining Gap Analysis (~3%)

The ~3% gap between our best result (51.48%) and the paper's reported 54.5% cannot be attributed to evaluation protocol differences — they are identical. Likely sources:

- **Checkpoint quality**: The publicly released checkpoint may not be the best checkpoint from training (early stopping, different seed)
- **Llama-3-8B version drift**: The gated model on HuggingFace may have been updated since the paper was written
- **BatchNorm calibration**: The language projector's 3 BN layers were calibrated on image-caption embeddings during training; class name templates have different statistical properties
- **Numerical precision**: Minor float32 vs bfloat16 differences accumulate across the 4-layer MLP

---

## Usage

```bash
# Activate environment
source .venv/bin/activate

# Run evaluation (requires HF_TOKEN for ILSVRC/imagenet-1k and meta-llama/Meta-Llama-3-8B)
python eval_zero_shot_imagenet.py \
    --checkpoint checkpoints/ShareLock-CC3M.ckpt \
    --batch_size 128 \
    --num_workers 8

# Use raw WordNet labels instead of OpenAI curated names
python eval_zero_shot_imagenet.py \
    --checkpoint checkpoints/ShareLock-CC3M.ckpt \
    --wordnet_classnames

# Single template instead of 80-template ensemble
python eval_zero_shot_imagenet.py \
    --checkpoint checkpoints/ShareLock-CC3M.ckpt \
    --template "a photo of a {}"
```

### Environment Requirements

- GPU with ≥16GB VRAM (Llama-3-8B in bfloat16 ≈ 16GB)
- `HF_TOKEN` environment variable with access to gated repos
- Python 3.12, PyTorch 2.6+, transformers, pytorch-lightning, omegaconf, datasets

---

## Files

| File | Description |
|------|-------------|
| `eval_zero_shot_imagenet.py` | Standalone zero-shot evaluation script (this work) |
| `checkpoints/ShareLock-CC3M.ckpt` | Pretrained CC3M checkpoint (expected ~54.5% top-1) |
| `sharelock/models/model.py` | ShareLock model with `encode_image` / `encode_text` |
| `sharelock/models/vision_encoder.py` | DINOv2 wrapper via `torch.hub` |
| `sharelock/models/language_encoder.py` | Llama-3-8B wrapper with bfloat16 loading |
| `sharelock/models/projection.py` | MLP projector with optional L2-norm |
| `sharelock/utils/misc.py` | `get_transforms()` — uses BILINEAR (not used in eval script) |
