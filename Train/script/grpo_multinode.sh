set -x

export DS_ENV_FILE="Train/config/deepspeed_env"



# your sft checkpoint
model="Train/output/sft"


# please ref https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node
hostfile="your hostfile path"

nproc_per_node=8
ts=`date +%Y%m%d-%H%M%S`
EXP_NAME_PREFIX="grpo"

# your grpo data path
DATASET="Data/grpo.jsonl"
OUTPUT_DIR="Train/output/grpo"
LOG_DIR="Train/log"

GRPO_CLI_PATH="ThirdPartyLib/ms-swift/swift/cli/rlhf.py"



deepspeed --hostfile $hostfile ${GRPO_CLI_PATH} \
        --rlhf_type grpo \
        --model $model \
        --reward_funcs external_timestamp_format_reward external_timestamp_length_reward external_dense_caption_f1_reward external_dense_caption_sodam_reward \
        --reward_weights 0.5 0.5 1.0 1.0 \
        --train_type full \
        --torch_dtype bfloat16 \
        --dataset ${DATASET} \
        --external_plugins ThirdPartyLib/ms-swift/examples/train/grpo/plugin/plugin.py  \
        --max_completion_length 9216 \
        --max_length 32768 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --learning_rate 1e-5 \
        --freeze_vit true \
        --freeze_aligner false \
        --gradient_accumulation_steps 1 \
        --beta 0.04 \
        --eval_steps 100 \
        --save_steps 10 \
        --save_only_model true \
        --save_total_limit 20 \
        --logging_steps 1 \
        --remove_unused_columns false \
        --output_dir "${OUTPUT_DIR}" \
        --logging_dir "${LOG_DIR}" \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 4 \
        --dataset_num_proc 4 \
        --num_generations 8 \
        --temperature 1. \
        --top_p 0.99 \
        --top_k 50 \
        --deepspeed zero2 \
        --attn_impl flash_attn \
        --log_entropy \
        --log_completions true \
        --report_to tensorboard 2>&1 | grep --line-buffered -v "Unused or unrecognized kwargs" | grep --line-buffered -v -e "UserWarning: PySoundFile failed. Trying audioread instead." -e "FutureWarning: librosa.core.audio.__audioread_load" -e "Deprecated as of librosa version 0.10.0." -e "It will be removed in librosa version 1.0." -e "y, sr_native = __audioread_load(path, offset, duration, dtype)" -e "librosa.load(video, sr=self.sampling_rate)" |  tee Train/log/${EXP_NAME}_${ts}.log