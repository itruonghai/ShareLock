# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ShareLock is a research project implementing an ultra-lightweight CLIP-like vision-language model. It uses **frozen** pretrained unimodal encoders (DINOv2 for vision, Llama-3-8B for language) with small **learnable projection networks** to align embeddings into a shared space via contrastive learning.
## Environment Setup

To set up the environment for ShareLock, run:

```bash
conda create -n sharelock python=3.12
conda activate sharelock
pip install -r requirements.txt
```

This will create a new Conda environment named `sharelock` with Python 3.12 and install all required dependencies.

For faster HuggingFace downloads (datasets and models), install `hf_transfer` (included in `requirements.txt`). The feature extraction script auto-enables it when the package is present.

```bash
pip install hf_transfer
```


## Before Running (Environment already installed)

```bash
conda activate sharelock
export HF_TOKEN=<your_huggingface_token>      # required for gated models (Llama-3-8B)
```

## Commands

### Download datasets
```bash
# CC3M training data
huggingface-cli download pingzhili/llava-filtered-cc3m-595k --repo-type dataset --local-dir datasets/

# ImageNet-1k validation set for evaluation (~6.5 GB)
huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset \
    --include 'data/validation*' --local-dir datasets/imagenet-1k
```

### Feature Precomputation (Required Before Training)

Before training, you must precompute frozen features for both vision and language encoders. This step is **mandatory**.

```bash
# Single GPU — extracts both vision and language features
python precompute_features.py \
  --hf_dataset pingzhili/llava-filtered-cc3m-595k \
  --vision_model dinov2_vitl14 \
  --language_model meta-llama/Meta-Llama-3-8B \
  --output_dir precomputed_features

# Multi-GPU — automatic sharding across 3 GPUs
python precompute_features.py \
  --vision_model dinov2_vitl14 \
  --language_model meta-llama/Meta-Llama-3-8B \
  --output_dir precomputed_features \
  --num_gpus 3

# Extract only one modality (e.g., vision already done)
python precompute_features.py --extract language \
  --language_model meta-llama/Meta-Llama-3-8B \
  --output_dir precomputed_features --num_gpus 3

# Use a specific GPU
python precompute_features.py --num_gpus 1 --gpu_id 2 ...
```

The script loads the HuggingFace dataset once (cached to `./datasets` by default via `--dataset_dir`), extracts vision features (DINOv2), unloads the vision model, then extracts language features (Llama-3-8B). With `--extract language`, only text is streamed (no image download). Multi-GPU uses `torch.multiprocessing.spawn` to shard work automatically. `hf_transfer` is auto-enabled when installed for faster downloads.

**Note:** All precomputed features (and metadata) are saved in subfolders of `precomputed_features/` and reused by the training pipeline.

### Training
```bash
# Single GPU
python train.py --config configs/cc3m_llava_config.yaml

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1 python train.py --config configs/cc3m_llava_config.yaml training.num_gpus=2

# Full pipeline (train + auto-eval)
./run_experiment.sh --config configs/cc3m_llava_config.yaml --gpu 0
```

### Evaluation (zero-shot ImageNet)
Requires the ImageNet-1k validation Parquet files (see "Download datasets" above).

```bash
python eval_zero_shot_imagenet.py \
    --checkpoint logs/<experiment>/version_X/checkpoints/best_model.ckpt \
    --config configs/cc3m_llava_config.yaml

# Custom Parquet path (default: datasets/imagenet-1k)
python eval_zero_shot_imagenet.py \
    --checkpoint checkpoints/ShareLock-CC3M.ckpt \
    --imagenet_data_dir /path/to/imagenet-1k

# Eval-only from run_experiment.sh
./run_experiment.sh --eval-only checkpoints/ShareLock-CC3M.ckpt --config configs/cc3m_llava_config.yaml
```

### Monitoring
```bash
tensorboard --logdir logs/ --port 6006
```

## Architecture

### Data Flow
1. Precompute and cache frozen encoder features (vision: 768-dim DINOv2, language: 4096-dim Llama)
2. During training, load cached features → learnable projectors → contrastive CLIP loss
3. At inference, encode via frozen encoders + trained projectors → L2-normalized 768-dim embeddings

### Key Components

| Component | File | Role |
|---|---|---|
| Main model | `sharelock/models/model.py` | `ShareLockModel` (LightningModule), loss, optimizer |
| Vision encoder | `sharelock/models/vision_encoder.py` | Frozen DINOv2/DINO/CLIP wrapper |
| Language encoder | `sharelock/models/language_encoder.py` | Frozen HuggingFace model wrapper (last token) |
| Projectors | `sharelock/models/projection.py` | MLP, MLPv2, QFormer, NVFormer variants |
| DataModule | `sharelock/data/data.py` | Lightning DataModule loading precomputed tensors |
| Datasets | `sharelock/data/datasets.py` | Vision [N,768] + Language [N,4096] feature tensors |
| Feature extraction | `precompute_features.py` | Unified vision+language precomputation with multi-GPU |

### Projection Variants (language projector: 4096→768)
- **MLP** (best, ~49.95% top-1): 4-layer Linear→BatchNorm→ReLU→Dropout, hidden=4096
- **MLPv2**: 4-layer with LayerNorm+GELU+residuals
- **QFormer**: 2-layer transformer with 32 learned queries
- **NVFormer**: Single cross-attention layer with 32 learned queries

Vision projector is Identity + L2-norm (no trainable parameters).

### Configuration System
Configs use OmegaConf YAML format. The best-performing config is `configs/cc3m_llava_config.yaml`. Config keys can be overridden from the command line (e.g., `training.num_gpus=2`).

### Training Details
- Framework: PyTorch Lightning with DDP multi-GPU support
- Loss: InfoNCE (CLIP contrastive loss) with learnable temperature (`logit_scale`)
- Optimizer: Adam with cosine LR schedule and warmup
- Data: 555k CC3M training samples, batches preloaded into memory
- Eval: Zero-shot ImageNet classification using 80 CLIP prompt templates

---

## EgoExo4D Video-Text Retrieval

Extension for egocentric video-text retrieval on the EgoExo4D dataset (atomic action descriptions).

### Prerequisites

```
EgoExo/
  takes.json                              # take metadata (fps, activity, etc.)
  annotations/atomic_descriptions_val.json
  val_takes.txt                           # newline-separated take_uids for val split
  val_videos/takes/<take_uid>/...         # ego-view MP4 files
```

### Step 1 — Precompute Video Features

```bash
# Single GPU, val split (adaptive 4-frame sampling around annotation timestamp)
python precompute_video_features.py \
    --annotation_json EgoExo/annotations/atomic_descriptions_val.json \
    --takes_json EgoExo/takes.json \
    --video_root EgoExo/val_videos/takes \
    --split_file EgoExo/val_takes.txt \
    --output_dir precomputed_features_video_adaptive_val \
    --vision_model vjepa2_vitl          # or egovlpv2

# Multi-GPU (shard across 4 GPUs)
python precompute_video_features.py ... --num_gpus 4

# Precompute language features (text encoder side)
python precompute_video_features.py \
    --annotation_json EgoExo/annotations/atomic_descriptions_val.json \
    --takes_json EgoExo/takes.json \
    --video_root EgoExo/val_videos/takes \
    --output_dir precomputed_features_language_val \
    --extract language \
    --language_model egovlpv2           # or meta-llama/Meta-Llama-3-8B
```

### Step 2 — Evaluate Retrieval

```bash
python eval_egoexo4d_retrieval.py \
    --checkpoint logs/<exp>/version_X/checkpoints/best_model.ckpt \
    --config configs/egoexo4d_vjepa2_config.yaml \
    --precomputed_features_dir precomputed_features_video_adaptive_val \
    --takes_json EgoExo/takes.json
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--max_eval_pairs` | 1000 | Subsample N pairs for faster eval (use 0 for full set) |
| `--per_category` | off | Print R@K / MCQ breakdown per activity category |
| `--save_plot` | off | Save 2-panel rank+score PNG to `<checkpoint>_analysis.png` |
| `--mcq_5way` | off | Run 5-way MCQ retrieval (1 GT + 4 same-take hard negatives) |
| `--annotation_json` | — | Required for `--mcq_5way`; path to atomic_descriptions JSON |
| `--video_root` | — | Required for `--mcq_5way`; path to raw video directory |
| `--split_file` | — | Optional val split filter for MCQ caption map |
| `--mcq_seed` | 42 | RNG seed for MCQ distractor sampling |

**Full example with MCQ and per-category breakdown:**

```bash
python eval_egoexo4d_retrieval.py \
    --checkpoint logs/egoexo4d_vjepa2.1_vitl/version_9/checkpoints/best_model.ckpt \
    --config configs/egoexo4d_vjepa2_config.yaml \
    --precomputed_features_dir precomputed_features_video_adaptive_val \
    --takes_json EgoExo/takes.json \
    --annotation_json EgoExo/annotations/atomic_descriptions_val.json \
    --video_root EgoExo/val_videos/takes \
    --split_file EgoExo/val_takes.txt \
    --max_eval_pairs 0 \
    --per_category \
    --save_plot \
    --mcq_5way
```

**Metrics reported:**

| Metric | Random baseline | Direction |
|--------|----------------|-----------|
| R@1, R@5, R@10 | 0.1% (N=1000) | V→T and T→V |
| MedR | N/2 ≈ 500 | V→T and T→V |
| MeanR | N/2 ≈ 500 | V→T and T→V |
| ScoreGap | 0.0 | overall |
| GT-pct | 50% | overall |
| MCQ 5-way accuracy | 20% | V→T |

### Step 3 — Visualize Retrieval Results

```bash
python visualize_retrieval.py \
    --checkpoint logs/<exp>/version_X/checkpoints/best_model.ckpt \
    --config configs/egoexo4d_vjepa2_config.yaml \
    --precomputed_features_dir precomputed_features_video_adaptive_val \
    --takes_json EgoExo/takes.json \
    --video_root EgoExo/val_videos/takes \
    --output retrieval_viz.mp4
```

**Direction (`--direction`):**

| Value | Description |
|-------|-------------|
| `v2t` | Video query → retrieve top-K captions (default) |
| `t2v` | Text query → retrieve top-K video clips (4 frame thumbnails each) |
| `both` | Side-by-side V→T and T→V for each query |
| `mcq` | 5-way MCQ: video + 5 labeled options A–E with cosine scores |

**Mode (`--mode`) — which queries to sample:**

| Value | Description |
|-------|-------------|
| `random` | Random queries from the pool (default) |
| `correct` | Queries where R@1 is correct |
| `incorrect` | Queries where R@1 is wrong |
| `mixed` | Mix of correct and incorrect (biased toward R@1 hits) |
| `per_category` | Sample `--queries_per_category` from each activity category |

**Other flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--num_queries` | 20 | Number of query slides to render |
| `--top_k` | 5 | Number of retrieved results shown per slide |
| `--queries_per_category` | 3 | Used with `--mode per_category` |
| `--mcq_seed` | 42 | RNG seed for MCQ option shuffling |
| `--annotation_json` | — | Required for `--direction mcq` |
| `--split_file` | — | Optional val split filter |

**MCQ visualization example:**

```bash
python visualize_retrieval.py \
    --checkpoint logs/egoexo4d_vjepa2.1_vitl/version_9/checkpoints/best_model.ckpt \
    --config configs/egoexo4d_vjepa2_config.yaml \
    --precomputed_features_dir precomputed_features_video_adaptive_val \
    --takes_json EgoExo/takes.json \
    --video_root EgoExo/val_videos/takes \
    --annotation_json EgoExo/annotations/atomic_descriptions_val.json \
    --split_file EgoExo/val_takes.txt \
    --direction mcq \
    --mode random \
    --num_queries 30 \
    --output retrieval_mcq.mp4
```

Each MCQ slide shows: top 38% = video frames, bottom 62% = 5 option boxes (green=GT, red=wrong model pick) with cosine similarity score badge and normalized bar chart per option.

### Notes on Results

- **R@1 is low (~4–5%)** because EgoExo4D atomic actions are semantically near-duplicate ("C picks up X", "C places Y"). The model still works: GT-pct ≈ 87% means the correct pair beats 87% of all distractors.
- **MCQ accuracy** (~37% at N=17k full set) is significantly above random (20%), confirming real discriminative ability. MCQ mitigates semantic collision by selecting distractors with a different primary action verb.
- **Negative cosine scores** are expected for non-GT pairs. InfoNCE loss pushes non-matching pairs apart on the unit hypersphere; background distribution centers near 0 with many negative values.
- See `analysis_egoexo4d.md` for full metric explanations and LLaMA-3-8B vs EgoVLPv2 comparison.
