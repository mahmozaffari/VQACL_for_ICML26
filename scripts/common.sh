#!/bin/bash
# Shared setup for the run scripts.
# Arguments: DATASET MODEL PARTITION [debug] [RESUME_FROM] [RESUME_TASK_IDX]

DATASET=${1:-vqa2}       # vqa2 or tdiuc
MODEL=${2:-vilt}         # vilt or flava
PARTITION=${3:-Q}        # Q: question-type tasks; V: TDIUC visual-category tasks
DEBUG_MODE=${4:-""}      # "debug" trains on a small subset
RESUME_FROM=${5:-""}     # run directory to resume from
RESUME_TASK_IDX=${6:-""} # task index to resume from (auto-detected if empty)

if [[ ! "$DATASET" =~ ^(vqa2|tdiuc)$ ]]; then echo "Error: dataset must be 'vqa2' or 'tdiuc'"; exit 1; fi
if [[ ! "$MODEL" =~ ^(vilt|flava)$ ]]; then echo "Error: model must be 'vilt' or 'flava'"; exit 1; fi
if [[ ! "$PARTITION" =~ ^(Q|V)$ ]]; then echo "Error: partition must be 'Q' or 'V'"; exit 1; fi
if [[ "$DATASET" == "vqa2" && "$PARTITION" != "Q" ]]; then echo "Error: VQA v2 uses partition 'Q'"; exit 1; fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."
source config/paths.sh

# Task sequences
if [[ "$DATASET" == "vqa2" ]]; then
    tasks=(recognition location judge commonsense count action color type subcategory causal)
    declare -A SHORT_TASK_NAMES=(
        [recognition]=re [location]=lo [judge]=ju [commonsense]=com [count]=cnt
        [action]=act [color]=col [type]=typ [subcategory]=sub [causal]=cau
    )
    TASK_PREFIX="q_"
elif [[ "$PARTITION" == "Q" ]]; then
    tasks=(color scene_recognition object_recognition counting positional_reasoning)
    declare -A SHORT_TASK_NAMES=(
        [color]=cr [scene_recognition]=sr [object_recognition]=or
        [counting]=ct [positional_reasoning]=pr
    )
    TASK_PREFIX=""
else
    tasks=(animal food indoor_activity outdoor_activity traffic)
    declare -A SHORT_TASK_NAMES=(
        [animal]=a [food]=f [indoor_activity]=ic
        [outdoor_activity]=oc [traffic]=t
    )
    TASK_PREFIX=""
fi

tasks_seq_dir=""
cl_task=""
for task in "${tasks[@]}"; do
    tasks_seq_dir="${tasks_seq_dir}${SHORT_TASK_NAMES[$task]}_"
    cl_task="${cl_task} ${TASK_PREFIX}${task}"
done
tasks_seq_dir=${tasks_seq_dir%_}

# Dataset paths and splits
if [[ "$DATASET" == "vqa2" ]]; then
    H5_PATH="${H5_BASE}/coco"
    VQA_DIR="${VQA_BASE}/vqa"
    TRAIN_SPLIT="karpathy_train"
    VAL_SPLIT="karpathy_val"
    TEST_SPLIT="karpathy_test"
    DATASET_ARG=""
    EXTRA_ARGS=""
    DATASET_DIR="vqa2"
else
    H5_PATH="${H5_BASE}/tdiuc"
    VQA_DIR="${VQA_BASE}/tdiuc"
    TRAIN_SPLIT="qian_train"
    VAL_SPLIT="qian_train"
    TEST_SPLIT="qian_val"
    DATASET_ARG="--dataset tdiuc"
    EXTRA_ARGS="--skip_validation"
    DATASET_DIR="tdiuc-${PARTITION}"
fi

# Resume
resume_args=""
RESUME_SUFFIX=""
if [[ -n "$RESUME_FROM" ]]; then
    resume_args="--resume_from $RESUME_FROM"
    RESUME_SUFFIX="RESUME_"
    if [[ -n "$RESUME_TASK_IDX" ]]; then
        resume_args="${resume_args} --resume_task_idx $RESUME_TASK_IDX"
        RESUME_SUFFIX="RESUME_T${RESUME_TASK_IDX}_"
    fi
fi

# Expert-training preset, chosen with the CONFIG environment variable:
#   release (default): 10 epochs, mixed precision, 50 warmup steps, early stopping. The
#            settings the scripts shipped with.
#   paper:   the settings of the runs behind the paper's tables: 20 epochs, 100 warmup steps,
#            LoRA alpha 32; ViLT in full precision with 2-step gradient accumulation and no
#            early stopping; FLAVA in mixed precision with an initial grad-scaler of 128
#            (gradient accumulation 2 on VQA v2; 1, with early stopping, on TDIUC).
#   sweep:   a later sweep that improved every table: 10 epochs, LoRA alpha 8 (equal to the
#            rank), warmup over the first 10% of each task's steps, the summed VQA loss
#            (--scale_vqa_loss), mixed precision, early stopping, batch 80 for ViLT.
# The preset also fixes the LoRA alpha that the router scripts load the experts with, so use
# the same CONFIG for stage 2. Runs of the paper and sweep presets are written under config
# folders prefixed paper_ and sweep_.
CONFIG="${CONFIG:-release}"
ES_ARGS="--early_stopping --early_stopping_patience 5 --early_stopping_delta 0.001"
case "$CONFIG" in
    release)
        EXP_EPOCHS=10; EXP_WARMUP=50; EXP_GRAD_ACC=1; LORA_ALPHA=32
        if [[ "$MODEL" == "vilt" ]]; then EXP_BS=128; else EXP_BS=32; fi
        EXP_AMP_ARGS="--use_amp"; EXP_ES_ARGS="$ES_ARGS"; EXP_LOSS_ARGS=""
        CONFIG_PREFIX=""; EXP_TAG="AMP"
        ;;
    paper)
        EXP_EPOCHS=20; EXP_WARMUP=100; LORA_ALPHA=32; EXP_LOSS_ARGS=""; CONFIG_PREFIX="paper_"
        if [[ "$MODEL" == "vilt" ]]; then
            EXP_BS=128; EXP_GRAD_ACC=2; EXP_AMP_ARGS=""; EXP_ES_ARGS=""; EXP_TAG="NoAMP_GA2"
        elif [[ "$DATASET" == "vqa2" ]]; then
            EXP_BS=32; EXP_GRAD_ACC=2; EXP_AMP_ARGS="--use_amp --init_grad_scaler 128"; EXP_ES_ARGS=""; EXP_TAG="AMP_GS128_GA2"
        else
            EXP_BS=32; EXP_GRAD_ACC=1; EXP_AMP_ARGS="--use_amp --init_grad_scaler 128"; EXP_ES_ARGS="$ES_ARGS"; EXP_TAG="AMP_GS128_ES"
        fi
        ;;
    sweep)
        # The trainer caps warmup at 10% of a task's steps, so a large value gives a 10% warmup.
        EXP_EPOCHS=10; EXP_WARMUP=100000000; EXP_GRAD_ACC=1; LORA_ALPHA=8
        if [[ "$MODEL" == "vilt" ]]; then EXP_BS=80; else EXP_BS=32; fi
        EXP_AMP_ARGS="--use_amp"; EXP_ES_ARGS="$ES_ARGS"; EXP_LOSS_ARGS="--scale_vqa_loss"
        CONFIG_PREFIX="sweep_"; EXP_TAG="a8_AMP_LS_WR0.1"
        ;;
    *)
        echo "Error: CONFIG must be release, paper, or sweep (got '$CONFIG')"; exit 1
        ;;
esac
