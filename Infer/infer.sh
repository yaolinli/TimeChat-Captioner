#!/bin/bash
# ==============================================================================
# Multi-GPU Inference Script for TimeChat-Captioner
#
# This script splits the evaluation dataset across multiple GPUs and runs
# inference in parallel.  After all workers finish, it merges the per-rank
# output files into a single JSONL file.
#
# Usage:
#   bash infer.sh
#
# Before running, make sure you have:
#   1. Downloaded & organized the OmniDCBench videos (see Infer/readme.md).
#   2. Adjusted the variables below to match your setup.
# ==============================================================================
set -e  # Exit immediately on error

# ==============================================================================
# User-configurable variables  (modify these to match your setup)
# ==============================================================================

# Path to the model checkpoint (local path or HuggingFace model ID)
MODEL_PATH="/mnt/user-ssd/yaolinli/code/TimeChat-Captioner/ckpts/TimeChat-Captioner-GRPO-7B"

OMNI_DCBENCH_DIR="/mnt/user-ssd/yaolinli/dataset-bosmm/OmniDCBench"
# Ground-truth JSONL file  (each line is a JSON object with a "clip_path" field)
INPUT_PATH="${OMNI_DCBENCH_DIR}/ours_gt_file.json"

# Root directory containing the video files (Movie/, Youtube/, etc.)
VIDEO_DIR="${OMNI_DCBENCH_DIR}/Video"

# Output directory for prediction files
OUTPUT_DIR="result/TimeChat-Captioner-GRPO-7B_infer"

# Number of GPUs to use
GPU_NUM=8

# The first GPU index (e.g., 0 means GPU 0,1,...,GPU_NUM-1)
GPU_START=0

# ==============================================================================
# Automatic setup  (no need to modify below)
# ==============================================================================

# Count total samples in the input file
DATA_LEN=$(wc -l < "$INPUT_PATH")
OFFSET=$((DATA_LEN / GPU_NUM))

echo "=============================="
echo "  TimeChat-Captioner Inference"
echo "=============================="
echo "  Model:       $MODEL_PATH"
echo "  Input:       $INPUT_PATH ($DATA_LEN samples)"
echo "  Video dir:   $VIDEO_DIR"
echo "  Output dir:  $OUTPUT_DIR"
echo "  GPUs:        $GPU_NUM (starting from GPU $GPU_START)"
echo "  Per-GPU:     ~$OFFSET samples"
echo "=============================="

# Create output and log directories
mkdir -p "$OUTPUT_DIR/logs"

# ==============================================================================
# Launch parallel inference workers (one per GPU)
# ==============================================================================
for ((rank=0; rank<GPU_NUM; rank++)); do
    start_index=$((rank * OFFSET))

    # The last worker takes all remaining samples
    if [ $rank -eq $((GPU_NUM - 1)) ]; then
        end_index=$DATA_LEN
    else
        end_index=$(((rank + 1) * OFFSET))
    fi

    GPU_ID=$((rank + GPU_START))
    LOG_FILE="${OUTPUT_DIR}/logs/gpu${rank}.log"

    echo "  Launching GPU $GPU_ID (rank $rank): samples [$start_index, $end_index)"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python inference.py \
        --model_path "$MODEL_PATH" \
        --input_path "$INPUT_PATH" \
        --video_dir "$VIDEO_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --start_index "$start_index" \
        --end_index "$end_index" \
        --rank "$rank" \
        > "$LOG_FILE" 2>&1 &
done

echo ""
echo "All workers launched. Waiting for completion..."
wait
echo "All workers finished."

# ==============================================================================
# Merge per-rank results into a single file
# ==============================================================================
MERGED_FILE="${OUTPUT_DIR}/merged_result.jsonl"

# Use a glob that works when run from any cwd
RANK_FILES=$(echo ${OUTPUT_DIR}/rank_*.jsonl)
if ! ls $RANK_FILES 1>/dev/null 2>&1; then
    echo "ERROR: No rank_*.jsonl files found in $OUTPUT_DIR"
    echo "Workers may have failed. Check logs: $OUTPUT_DIR/logs/gpu*.log"
    exit 1
fi

echo "Merging results into: $MERGED_FILE"
cat ${OUTPUT_DIR}/rank_*.jsonl > "$MERGED_FILE"
echo "Done! Total predictions: $(wc -l < "$MERGED_FILE")"
echo "Results saved to: $MERGED_FILE"
