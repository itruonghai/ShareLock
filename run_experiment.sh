#!/bin/bash
# run_experiment.sh — train then evaluate in one go
#
# Usage:
#   ./run_experiment.sh --config configs/cc3m_llava_config.yaml [--gpu 1] [--eval-only PATH]
#
# Examples:
#   ./run_experiment.sh --config configs/cc3m_qformer.yaml --gpu 0
#   ./run_experiment.sh --config configs/cc3m_llava_config.yaml --gpu 1 --eval-batch-size 1024
#   ./run_experiment.sh --eval-only logs/cc3m_llava/version_3/checkpoints/best.ckpt --config configs/cc3m_llava_config.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
GPU=0
CONFIG=""
EVAL_ONLY=""
EVAL_BATCH_SIZE=512
EVAL_WORKERS=8
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";          shift 2 ;;
        --gpu)          GPU="$2";             shift 2 ;;
        --eval-only)    EVAL_ONLY="$2";       shift 2 ;;
        --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --eval-workers) EVAL_WORKERS="$2";    shift 2 ;;
        *)              EXTRA_ARGS+=("$1");   shift ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    echo "Usage: $0 --config CONFIG_FILE [--gpu N] [--eval-only CHECKPOINT]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export TMPDIR="${TMPDIR:-/tmp/sharelock_staging}"
mkdir -p "$TMPDIR"

cd "$SCRIPT_DIR"

# ── Eval-only mode ──────────────────────────────────────────────────────────
if [[ -n "$EVAL_ONLY" ]]; then
    echo "==> Evaluating checkpoint: $EVAL_ONLY"
    "$PYTHON" eval_zero_shot_imagenet.py \
        --checkpoint "$EVAL_ONLY" \
        --config "$CONFIG" \
        --batch_size "$EVAL_BATCH_SIZE" \
        --num_workers "$EVAL_WORKERS"
    exit $?
fi

# ── Train ───────────────────────────────────────────────────────────────────
echo "==> Training  config=$CONFIG  gpu=$GPU"
TRAIN_LOG=$(mktemp /tmp/sharelock_train_XXXXX.log)
echo "    log → $TRAIN_LOG"

"$PYTHON" train.py \
    --config "$CONFIG" \
    training.num_gpus=1 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}
if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "==> Training failed (exit $TRAIN_EXIT)" >&2
    exit $TRAIN_EXIT
fi

# ── Extract best checkpoint path printed by train.py ───────────────────────
BEST_CKPT=$(grep -o 'BEST_CHECKPOINT=.*' "$TRAIN_LOG" | tail -1 | cut -d= -f2-)

if [[ -z "$BEST_CKPT" ]]; then
    echo "==> Warning: could not find BEST_CHECKPOINT in training log, skipping eval"
    exit 0
fi

echo ""
echo "==> Training complete. Best checkpoint: $BEST_CKPT"
echo "==> Running zero-shot ImageNet evaluation..."

"$PYTHON" eval_zero_shot_imagenet.py \
    --checkpoint "$BEST_CKPT" \
    --config "$CONFIG" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --num_workers "$EVAL_WORKERS" \

echo "==> Done."
