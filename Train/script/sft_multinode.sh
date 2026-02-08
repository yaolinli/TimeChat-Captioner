set -x

# please need to add your multi-node communication configuration to the env file.
export DS_ENV_FILE="Train/config/deepspeed_env"

model="Qwen/Qwen2.5-Omni-7B"

# please ref https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node
hostfile="your hostfile path"
nproc_per_node=8
ts=`date +%Y%m%d-%H%M%S`
EXP_NAME="sft_multinode"


SFT_CLI_PATH="ThirdPartyLib/ms-swift/swift/cli/sft.py"

# your sft data path
DATASET="Data/sft.jsonl"
OUTPUT_DIR="Train/output/sft"

unset NCCL_TOPO_FILE

deepspeed --hostfile $hostfile ${SFT_CLI_PATH} \
    --model $model  \
    --save_only_model \
    --load_from_cache_file False \
    --save_strategy epoch \
    --dataset ${DATASET} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --freeze_vit true \
    --gradient_accumulation_steps 2 \
    --eval_steps 200000 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --split_dataset_ratio 0 \
    --logging_steps 5 \
    --max_length 32768 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --train_type full \
    --torch_dtype bfloat16 \
    --deepspeed zero2 \
    --truncation_strategy delete \
    --attn_impl flash_attn \
    --lr_scheduler_type "cosine" \
    --lr_scheduler_kwargs '{"num_cycles": 0.47}' \
    --report_to tensorboard 2>&1 | grep --line-buffered -v "Unused or unrecognized kwargs"| \
    grep --line-buffered -v "librosa.core.audio.__audioread_load" | \
    tee Train/log/${EXP_NAME}_${ts}.log