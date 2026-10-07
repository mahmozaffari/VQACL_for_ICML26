#!/bin/bash
# Ablation: router trained to predict the task ID (instead of expected accuracy)
# on top of the frozen stage-1 experts, then evaluate standard, Bayesian, and oracle routing.
# Usage: EXPERT_CKPT=<stage-1 run>/checkpoints/vqa \
#        bash scripts/train_taskid_router.sh DATASET MODEL PARTITION [debug] [RESUME_FROM] [RESUME_TASK_IDX]

source "$( dirname "${BASH_SOURCE[0]}" )/common.sh"

if [[ -z "$EXPERT_CKPT" ]]; then
    echo "Error: set EXPERT_CKPT to the checkpoints/vqa directory of a stage-1 run (scripts/train_experts.sh)"
    exit 1
fi

bs=128

debug_args=""
if [[ "$DEBUG_MODE" == "debug" ]]; then
    debug_args="--train_topk 5000"
    tasks_seq_dir="${tasks_seq_dir}_DEBUG"
fi

lora_r=8
lr=1e-3
epochs=20
router_lr=1e-3
router_epochs=5

config_str="${RESUME_SUFFIX}EP${epochs}_mlpEP${router_epochs}_LR${lr}_mlpLR${router_lr}_BS${bs}_LoRA${lora_r}_AMP"
output_dir="./experiments/${DATASET_DIR}/${MODEL}/taskid_router/${tasks_seq_dir}/${config_str}"

python src/main_train.py \
    --strategy moe_router \
    --training_mode mlp_only \
    $DATASET_ARG \
    --model_name $MODEL \
    --output $output_dir \
    --cl_tasks $cl_task \
    --epochs $epochs \
    --lr $lr \
    --mlp_epochs $router_epochs \
    --mlp_lr $router_lr \
    --use_lora \
    --lora_r $lora_r \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --test_split $TEST_SPLIT \
    --partition_name Partition_${PARTITION} \
    --classifier \
    --batch_size $bs \
    --clip_grad_norm 5 \
    --gradient_accumulation_steps 2 \
    --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --freeze_base \
    --use_h5 \
    --h5_path $H5_PATH \
    --vqa_dir $VQA_DIR \
    --now_train \
    $debug_args \
    --use_amp \
    --init_grad_scaler 128 \
    --vqa_checkpoint_path $EXPERT_CKPT \
    $resume_args \
    $EXTRA_ARGS
