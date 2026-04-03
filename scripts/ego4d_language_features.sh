#!/bin/bash
# =============================================================================
# Precompute Llama-3-8B language features for Ego4D egovid-5m
#
# Usage:
#   sbatch scripts/ego4d_language_features.sh            # train split (default)
#   SPLIT=val sbatch scripts/ego4d_language_features.sh  # val split
#
# Requires HF_TOKEN to be set (Llama-3-8B is a gated model):
#   export HF_TOKEN=hf_...
#   sbatch scripts/ego4d_language_features.sh
#
# Edit the USER CONFIG section below before submitting.
# =============================================================================

#SBATCH --job-name=ego4d_lang
#SBATCH --partition=gpu            # <-- change to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_lang.out
#SBATCH --error=logs/%j_lang.err

# =============================================================================
# USER CONFIG — edit these paths for your cluster
# =============================================================================
EGO4D_ROOT="/path/to/ego4d"
SHARELOCK_DIR="$HOME/ShareLock"
CONDA_ENV="sharelock"
OUTPUT_DIR="$SHARELOCK_DIR/precomputed_features_ego4d"

# Split selection: train (default) or val
SPLIT="${SPLIT:-train}"
if [ "$SPLIT" = "val" ]; then
    CSV_FILE="$EGO4D_ROOT/egovid-val.csv"
    OUTPUT_DIR="${OUTPUT_DIR}_val"
    CAPTION_NAME="egovid_val"
    JOB_TAG="val"
else
    CSV_FILE="$EGO4D_ROOT/egovid-text.csv"
    CAPTION_NAME="egovid_train"
    JOB_TAG="train"
fi
# =============================================================================

if [ -z "$HF_TOKEN" ]; then
    echo "[ERROR] HF_TOKEN is not set. Llama-3-8B is a gated model."
    echo "        Run: export HF_TOKEN=hf_... then resubmit."
    exit 1
fi

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Split        : $JOB_TAG"
echo "CSV          : $CSV_FILE"
echo "Caption name : $CAPTION_NAME"
echo "Output dir   : $OUTPUT_DIR"
echo "Started      : $(date)"
echo "========================================"

# ── Environment ──────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1   # faster HuggingFace downloads

cd "$SHARELOCK_DIR"

# ── Run ──────────────────────────────────────────────────────────────────────
python precompute_video_features.py \
    --dataset ego4d \
    --ego4d_root     "$EGO4D_ROOT" \
    --csv_file       "$CSV_FILE" \
    --language_model meta-llama/Meta-Llama-3-8B \
    --caption_name   "$CAPTION_NAME" \
    --output_dir     "$OUTPUT_DIR" \
    --extract        language \
    --num_gpus       1 \
    --language_batch_size 128

echo "========================================"
echo "Finished : $(date)"
echo "========================================"
