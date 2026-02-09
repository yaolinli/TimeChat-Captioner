#!/bin/bash
set -e  # 出错立即退出
set -x  # 打印执行的命令

################ 评测设置 #################

# 🎯 修改为 Hugging Face 模型 ID
MODEL_PATH="yaolily/TimeChat-Captioner-GRPO-7B"

# 📂 输出目录 (根据新模型命名)
BASE_OUTPUT_DIR="result"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/TimeChat-Captioner-GRPO-7B_infer"

# 💿 数据集路径 (保持不变)
INPUT_PATH="gt_file.jsonl"
VIDEO_DIR="videos"

# ⚙️ GPU 设置
GPU_NUM=8
CUDA_VISIBLE_DEVICES_BASE=0

# === 预先计算数据总行数 ===
DATA_LEN=$($PYTHON_BIN - <<'PYCODE'
import json
# 注意：这里硬编码了路径，确保与 INPUT_PATH 一致
path = "gt_file.jsonl"
count = sum(1 for _ in open(path, "r"))
print(count)
PYCODE
)

OFFSET=$((DATA_LEN / GPU_NUM))

echo "数据总条数: $DATA_LEN"
echo "每张卡分配 $OFFSET 条样本"

########################################
### 🚀 开始执行推理 ###
########################################

echo "=============================="
echo "⭐  开始评测模型: $MODEL_PATH"
echo "=============================="

# 创建日志目录
mkdir -p "$OUTPUT_DIR/logs"

# 注意：已移除修改 config.json 的代码块
# 因为直接使用 HF 模型 ID 时无法直接修改远程文件的配置。
# 如果必须修改 use_cache，请在 inference.py 中加载模型时设置，
# 或者先使用 huggingface-cli download 下载到本地后再修改。

# 启动多 GPU 推理
for ((rank=0; rank<GPU_NUM; rank++)); do
    start_index=$((rank * OFFSET))

    if [ $rank -eq $((GPU_NUM - 1)) ]; then
        end_index=$DATA_LEN
    else
        end_index=$(((rank + 1) * OFFSET))
    fi

    LOG_FILE="${OUTPUT_DIR}/logs/infer_gpu${rank}.log"

    echo "启动 GPU $rank: [$start_index, $end_index)"

    # 这里的 inference.py 需要支持 --model_path 传入 HF ID
    CUDA_VISIBLE_DEVICES=$((rank + CUDA_VISIBLE_DEVICES_BASE)) \
    $PYTHON_BIN /home/gaohuan03/yaolinli/code/qwen25omni/OmniVideoCaption/Inference/inference.py \
        --input_path "$INPUT_PATH" \
        --video_dir "$VIDEO_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_PATH" \
        --start_index "$start_index" \
        --end_index "$end_index" \
        --rank "$rank" \
        >"$LOG_FILE" 2>&1 &
done

wait
echo "✔ 推理任务全部完成"

# 合并结果
MERGED_FILE="${OUTPUT_DIR}/merged_result.jsonl"
echo "合并结果到 $MERGED_FILE"
cat ${OUTPUT_DIR}/*_*.jsonl > "$MERGED_FILE"

echo "🎉 评测完毕！结果已保存至: $MERGED_FILE"