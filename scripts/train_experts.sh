#!/bin/bash
# Stage 1: train one LoRA expert and answer head per task.
# Usage: bash scripts/train_experts.sh DATASET MODEL PARTITION [debug] [RESUME_FROM] [RESUME_TASK_IDX]

source "$( dirname "${BASH_SOURCE[0]}" )/common.sh"

if [[ "$MODEL" == "vilt" ]]; then
    bs=128
else
    bs=32
fi

debug_args=""
if [[ "$DEBUG_MODE" == "debug" ]]; then
    debug_args="--train_topk 5000"
    tasks_seq_dir="${tasks_seq_dir}_DEBUG"
fi

lora_r=8
lr=1e-3
epochs=10
warmup_steps=50

config_str="${RESUME_SUFFIX}EP${epochs}_LR${lr}_BS${bs}_LoRA${lora_r}_AMP"
output_dir="./experiments/${DATASET_DIR}/${MODEL}/experts/${tasks_seq_dir}/${config_str}"

python src/main_train.py \
    --strategy moe \
    $DATASET_ARG \
    --model_name $MODEL \
    --output $output_dir \
    --cl_tasks $cl_task \
    --epochs $epochs \
    --lr $lr \
    --use_lora \
    --lora_r $lora_r \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --test_split $TEST_SPLIT \
    --partition_name Partition_${PARTITION} \
    --classifier \
    --batch_size $bs \
    --clip_grad_norm 5 \
    --gradient_accumulation_steps 1 \
    --warmup_steps $warmup_steps \
    --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --freeze_base \
    --use_h5 \
    --h5_path $H5_PATH \
    --vqa_dir $VQA_DIR \
    --now_train \
    --early_stopping \
    --early_stopping_patience 5 \
    --early_stopping_delta 0.001 \
    $debug_args \
    --use_amp \
    $resume_args \
    $EXTRA_ARGS \
    --skip_bayesian_evaluation
