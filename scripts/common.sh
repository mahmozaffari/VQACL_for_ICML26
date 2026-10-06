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
