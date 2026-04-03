#!/bin/bash
# =============================================================================
# Precompute V-JEPA-2.1 ViT-iG video features for Ego4D egovid-5m
#
# Usage:
#   sbatch scripts/ego4d_video_features.sh            # train split (default)
#   SPLIT=val sbatch scripts/ego4d_video_features.sh  # val split
#
# Edit the USER CONFIG section below before submitting.
# =============================================================================

#SBATCH --job-name=ego4d_video
#SBATCH --partition=gpu            # <-- change to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j_video.out
#SBATCH --error=logs/%j_video.err

# =============================================================================
# USER CONFIG — edit these paths for your cluster
# =============================================================================
EGO4D_ROOT="/path/to/ego4d"              # root with v2/video_540ss/ inside
SHARELOCK_DIR="$HOME/ShareLock"          # repo root
CONDA_ENV="sharelock"
OUTPUT_DIR="$SHARELOCK_DIR/precomputed_features_ego4d"

# Split selection: train (default) or val
SPLIT="${SPLIT:-train}"
if [ "$SPLIT" = "val" ]; then
    CSV_FILE="$EGO4D_ROOT/egovid-val.csv"
    OUTPUT_DIR="${OUTPUT_DIR}_val"
    JOB_TAG="val"
else
    CSV_FILE="$EGO4D_ROOT/egovid-text.csv"
    JOB_TAG="train"
fi
# =============================================================================

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Split        : $JOB_TAG"
echo "CSV          : $CSV_FILE"
echo "Output dir   : $OUTPUT_DIR"
echo "Started      : $(date)"
echo "========================================"

# ── Environment ──────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$SHARELOCK_DIR"

# ── Run ──────────────────────────────────────────────────────────────────────
python precompute_video_features.py \
    --dataset ego4d \
    --ego4d_root "$EGO4D_ROOT" \
    --csv_file   "$CSV_FILE" \
    --variant    vjepa2.1_vitig_384 \
    --output_dir "$OUTPUT_DIR" \
    --extract    video \
    --num_gpus   1 \
    --batch_size 8 \
    --num_workers 16

echo "========================================"
echo "Finished : $(date)"
echo "========================================"
