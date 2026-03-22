#!/bin/bash
# run_all_evals.sh — run all checkpoints across 2 GPUs in parallel queues
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
LOGDIR="$SCRIPT_DIR/eval_logs"
mkdir -p "$LOGDIR"

cd "$SCRIPT_DIR"

# GPU 0 queue: CC3M (cached protos) → version_6 → version_1 → qformer/v3
gpu0_evals=(
    "checkpoints/ShareLock-CC3M.ckpt|configs/cc3m_llava_config.yaml|cc3m_published"
    "logs/cc3m_llava/version_6/checkpoints/best_model.ckpt|configs/cc3m_llava_config.yaml|llava_v6"
    "logs/cc3m_llava/version_1/checkpoints/best_model.ckpt|configs/cc3m_llava_config.yaml|llava_v1"
    "logs/cc3m_qformer/version_3/checkpoints/best_model.ckpt|configs/cc3m_qformer.yaml|qformer_v3"
)

# GPU 1 queue: version_7 (cached protos) → mlp_v2/v0 → mlp_v2/v1
gpu1_evals=(
    "logs/cc3m_llava/version_7/checkpoints/best_model.ckpt|configs/cc3m_llava_config.yaml|llava_v7"
    "logs/cc3m_mlp_v2/version_0/checkpoints/best_model.ckpt|configs/cc3m_mlp_v2.yaml|mlp_v2_v0"
    "logs/cc3m_mlp_v2/version_1/checkpoints/best_model.ckpt|configs/cc3m_mlp_v2.yaml|mlp_v2_v1"
)

run_queue() {
    local gpu="$1"
    shift
    local evals=("$@")
    export CUDA_VISIBLE_DEVICES="$gpu"
    for entry in "${evals[@]}"; do
        IFS='|' read -r ckpt config tag <<< "$entry"
        logfile="$LOGDIR/${tag}.log"
        echo "[GPU $gpu] Starting: $tag → $logfile"
        "$PYTHON" eval_zero_shot_imagenet.py \
            --checkpoint "$ckpt" \
            --config "$config" \
            --batch_size 512 \
            --num_workers 8 \
            2>&1 | tee "$logfile"
        # Extract and print result
        result=$(grep -E "Top-[15] Accuracy" "$logfile" | tail -2 || true)
        echo "[GPU $gpu] DONE $tag: $result"
    done
}

export -f run_queue

# Run both GPU queues in parallel
run_queue 0 "${gpu0_evals[@]}" &
PID0=$!
run_queue 1 "${gpu1_evals[@]}" &
PID1=$!

wait $PID0
wait $PID1

echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE — Summary"
echo "=========================================="
for logfile in "$LOGDIR"/*.log; do
    tag=$(basename "$logfile" .log)
    ckpt=$(grep "Checkpoint" "$logfile" | head -1 | sed 's/.*: //')
    top1=$(grep "Top-1 Accuracy" "$logfile" | tail -1 | sed 's/.*: //')
    top5=$(grep "Top-5 Accuracy" "$logfile" | tail -1 | sed 's/.*: //')
    echo "  $tag: top-1=$top1  top-5=$top5"
done
