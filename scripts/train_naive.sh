#!/bin/bash
# Baseline: sequential fine-tuning of a single model with one unified answer head.
# Usage: bash scripts/train_naive.sh DATASET MODEL PARTITION [debug] [RESUME_FROM] [RESUME_TASK_IDX]

source "$( dirname "${BASH_SOURCE[0]}" )/common.sh"

if [[ "$MODEL" == "vilt" ]]; then
    bs=256
else
    bs=32
fi

debug_args=""
if [[ "$DEBUG_MODE" == "debug" ]]; then
    debug_args="--train_topk 10000"
    tasks_seq_dir="${tasks_seq_dir}_DEBUG"
fi

lr=1e-4
epochs=20

config_str="${RESUME_SUFFIX}EP${epochs}_LR${lr}_BS${bs}_AMP"
output_dir="./experiments/${DATASET_DIR}/${MODEL}/naive/${tasks_seq_dir}/${config_str}"

python src/main_train.py \
    --strategy naive \
    $DATASET_ARG \
    --model_name $MODEL \
    --output $output_dir \
    --cl_tasks $cl_task \
    --epochs $epochs \
    --lr $lr \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --test_split $TEST_SPLIT \
    --partition_name Partition_${PARTITION} \
    --classifier \
    --batch_size $bs \
    --gradient_accumulation_steps 1 \
    --num_workers 2 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_h5 \
    --h5_path $H5_PATH \
    --vqa_dir $VQA_DIR \
    --now_train \
    $debug_args \
    --use_amp \
    --init_grad_scaler 128 \
    $resume_args \
    $EXTRA_ARGS
