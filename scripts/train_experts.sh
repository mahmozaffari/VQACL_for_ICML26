#!/bin/bash
# Stage 1: train one LoRA expert and answer head per task.
# Usage: [CONFIG=release|paper|sweep] bash scripts/train_experts.sh DATASET MODEL PARTITION [debug] [RESUME_FROM] [RESUME_TASK_IDX]

source "$( dirname "${BASH_SOURCE[0]}" )/common.sh"

debug_args=""
if [[ "$DEBUG_MODE" == "debug" ]]; then
    debug_args="--train_topk 5000"
    tasks_seq_dir="${tasks_seq_dir}_DEBUG"
fi

# Epochs, batch size, warmup, LoRA alpha, precision, early stopping and loss scaling come from
# the CONFIG preset in common.sh.
bs=$EXP_BS
lora_r=8
lora_alpha=$LORA_ALPHA
lr=1e-3
epochs=$EXP_EPOCHS
warmup_steps=$EXP_WARMUP

config_str="${RESUME_SUFFIX}${CONFIG_PREFIX}EP${epochs}_LR${lr}_BS${bs}_LoRA${lora_r}_${EXP_TAG}"
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
    --lora_alpha $lora_alpha \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --test_split $TEST_SPLIT \
    --partition_name Partition_${PARTITION} \
    --classifier \
    --batch_size $bs \
    --clip_grad_norm 5 \
    --gradient_accumulation_steps $EXP_GRAD_ACC \
    --warmup_steps $warmup_steps \
    --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --freeze_base \
    --use_h5 \
    --h5_path $H5_PATH \
    --vqa_dir $VQA_DIR \
    --now_train \
    $EXP_ES_ARGS \
    $EXP_LOSS_ARGS \
    $debug_args \
    $EXP_AMP_ARGS \
    $resume_args \
    $EXTRA_ARGS \
    --skip_bayesian_evaluation
