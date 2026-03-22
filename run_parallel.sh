#!/bin/bash
# Run two training experiments in parallel, one per GPU.
#
# Usage:
#   ./run_parallel.sh configs/exp1.yaml configs/exp2.yaml
#
# Each experiment gets its own GPU (CUDA_VISIBLE_DEVICES=0 / 1) and its own
# log file under logs/runs/. Progress is printed to both the file and stdout
# with a [EXP1] / [EXP2] prefix so you can watch both at once:
#
#   tail -f logs/runs/<timestamp>_exp1.log
#   tail -f logs/runs/<timestamp>_exp2.log
#   # or both simultaneously:
#   tail -f logs/runs/<timestamp>_exp*.log

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config1.yaml> <config2.yaml> [extra train.py args...]"
    exit 1
fi

CONFIG1="$1"
CONFIG2="$2"
shift 2
EXTRA_ARGS=("$@")

LOGDIR="logs/runs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

LOG1="$LOGDIR/${TIMESTAMP}_exp1.log"
LOG2="$LOGDIR/${TIMESTAMP}_exp2.log"

echo "============================================"
echo "  EXP1: $CONFIG1  →  GPU 0  →  $LOG1"
echo "  EXP2: $CONFIG2  →  GPU 1  →  $LOG2"
echo "  Watch logs:"
echo "    tail -f $LOG1"
echo "    tail -f $LOG2"
echo "============================================"

# PYTHONUNBUFFERED=1 ensures every print() reaches the file immediately
# training.num_gpus=1 overrides the config so each job uses only 1 GPU
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train.py \
    --config "$CONFIG1" \
    training.num_gpus=1 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG1" &
PID1=$!

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python train.py \
    --config "$CONFIG2" \
    training.num_gpus=1 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG2" &
PID2=$!

echo "Started — PIDs: exp1=$PID1, exp2=$PID2"
echo ""

# Wait for both and report exit codes
wait $PID1; RC1=$?
wait $PID2; RC2=$?

echo ""
echo "============================================"
[ $RC1 -eq 0 ] && echo "  EXP1 DONE   $CONFIG1" || echo "  EXP1 FAILED (exit $RC1)  $CONFIG1"
[ $RC2 -eq 0 ] && echo "  EXP2 DONE   $CONFIG2" || echo "  EXP2 FAILED (exit $RC2)  $CONFIG2"
echo "============================================"

[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ]
