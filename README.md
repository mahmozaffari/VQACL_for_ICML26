# Calibrated Knowledge Aggregation in Bayesian Mixture-of-Experts for Continual VQA

**Mahsa Mozaffari, Hitesh Sapkota, Yu Kong, Xumin Liu, Qi Yu**
ICML 2026 · [Paper](https://mahsamozaffari.com/wp-content/uploads/2026/05/ICML_2026.pdf)

> **Note:** The preprocessed data and trained checkpoints will be made available soon.

This repository contains the official PyTorch implementation of the paper.

## Overview

Continual VQA is usually handled by training one expert per task and routing each question with task-ID supervision. Continual VQA tasks overlap, so an expert trained on one task often answers questions from another task well, and hard routing to a single expert can be confidently wrong. This code implements the method from the paper:

- **Per-task experts.** Each task adds LoRA adapters (rank 8, alpha 32) to the feed-forward layers of a frozen ViLT or FLAVA backbone, together with its own answer head. Experts are frozen after their task.
- **Utility router.** A 2-layer MLP over BERT question embeddings is trained to maximize the expected VQA accuracy of the experts it weights. An entropy term keeps probability mass on several plausible experts instead of collapsing to one.
- **Bayesian aggregation.** At inference, the predictive distributions of the top-k experts (k = 3) are combined in a unified answer space rather than committing to a single expert.

## Repository layout

```text
config/paths.sh                     data locations used by the run scripts
scripts/train_experts.sh            stage 1: train the per-task experts
scripts/train_utility_router.sh     stage 2: train the utility router and evaluate (our method)
scripts/train_taskid_router.sh      ablation: router trained to predict the task ID
scripts/train_naive.sh              baseline: sequential fine-tuning with one shared head
src/main_train.py                   training and evaluation entry point
src/strategies/                     expert training (moe.py), routers, and the naive baseline
src/vqa_dataset.py                  VQA v2 and TDIUC datasets and loaders
tools/build_h5_cache.py             builds the image caches
tools/summarize_vqa2.py             summarizes accuracy, forgetting, and calibration of a run (VQA v2)
tools/summarize_tdiuc.py            the same for TDIUC
```

## Installation

The code was tested with Python 3.10, PyTorch 2.7.1 (CUDA 11.8), and Transformers 4.57.1 on NVIDIA RTX A6000 GPUs.

```bash
conda create -n cvqa python=3.10
conda activate cvqa
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Pick the PyTorch build that matches your CUDA driver. The pretrained models (`dandelin/vilt-b32-mlm`, `facebook/flava-full`, `bert-base-uncased`) are downloaded from Hugging Face on first use.

## Data

We evaluate on two benchmarks:

- **VQA v2**, split into the 10 question-type tasks of [VQACL](https://github.com/zhangxi1997/VQACL) (Zhang et al., CVPR 2023).
- **TDIUC**, under the two continual protocols of TRIPLET (Qian et al., ICCV 2023): CL-LS, with 5 question-type tasks (partition `Q`), and CL-VS, with 5 visual-category tasks (partition `V`).

The preprocessed task splits and answer vocabularies for both benchmarks will be made available soon. Once released, extract them into `datasets/` so that they match the layout below. The scripts read question and answer files from `datasets/` and image caches from `h5_dataset/`. Both locations can be changed in `config/paths.sh`.

```text
datasets/
├── vqa/
│   ├── v2_mscoco_train2014_annotations.json
│   ├── v2_mscoco_val2014_annotations.json
│   ├── karpathy_train.json, karpathy_val.json, karpathy_test.json
│   ├── trainval_ans2label.json, trainval_label2ans.json
│   └── Partition_Q/
│       ├── karpathy_{train,val,test}_q_<task>.json
│       └── q_<task>_ans2label.json
└── tdiuc/
    ├── mscoco_train2014_annotations.json, mscoco_val2014_annotations.json
    ├── qian_train.json, qian_val.json
    ├── train_ans2label.json, train_label2ans.json
    ├── Partition_Q/
    │   ├── qian_{train,val}_<task>.json
    │   └── <task>_ans2label.json
    └── Partition_V/
        ├── qian_{train,val}_<task>.json
        └── <task>_ans2label.json
```

The original data come from [VQA v2](https://visualqa.org/download.html), the [VQACL repository](https://github.com/zhangxi1997/VQACL) (Karpathy splits and VQA v2 task partitions), and the [TDIUC project page](https://kushalkafle.com/projects/tdiuc.html).

### Image caches

Images are read from H5 caches of 384 x 384 images. Build them from the COCO 2014 images (VQA v2) and from the TDIUC images, each with `train2014/` and `val2014/` subfolders:

```bash
python tools/build_h5_cache.py --cocoroot /path/to/coco/images  --split train --outdir h5_dataset/coco
python tools/build_h5_cache.py --cocoroot /path/to/coco/images  --split valid --outdir h5_dataset/coco
python tools/build_h5_cache.py --cocoroot /path/to/tdiuc/Images --split train --outdir h5_dataset/tdiuc
python tools/build_h5_cache.py --cocoroot /path/to/tdiuc/Images --split valid --outdir h5_dataset/tdiuc
```

The VQA v2 Karpathy test split is drawn from `val2014`, so `valid_384.h5` serves both validation and test.

## Training and evaluation

Run all commands from the repository root. Each script takes `DATASET MODEL PARTITION`, where the dataset is `vqa2` or `tdiuc`, the model is `vilt` or `flava`, and the partition is `Q` or `V` (VQA v2 uses `Q`).

**Stage 1: train the experts.** One LoRA expert and answer head is trained per task, in task order.

```bash
bash scripts/train_experts.sh vqa2 vilt Q
```

**Stage 2: train the utility router and evaluate.** Point `EXPERT_CKPT` at the `checkpoints/vqa` folder of the stage-1 run.

```bash
EXPERT_CKPT=experiments/vqa2/vilt/experts/<tasks>/<config>/<timestamp>_moe/checkpoints/vqa \
bash scripts/train_utility_router.sh vqa2 vilt Q
```

The task-ID router ablation (`train_taskid_router.sh`) takes the same `EXPERT_CKPT`. The naive baseline (`train_naive.sh`) trains on its own.

Add `debug` as a fourth argument to train on a small subset of each task, for example `bash scripts/train_experts.sh vqa2 vilt Q debug`. To resume an interrupted run, pass the run directory and, optionally, the task index as the fifth and sixth arguments.

### Hyperparameters

| | ViLT | FLAVA |
|---|---|---|
| LoRA rank / alpha | 8 / 32 | 8 / 32 |
| Expert training: epochs, learning rate, batch size | 10, 1e-3, 128 | 10, 1e-3, 32 |
| Router training: epochs, learning rate, batch size | 5, 1e-3, 128 | 5, 1e-3, 128 |
| Entropy weight | 0.1 | 0.1 |
| Router rehearsal memory | 5000 | 5000 |
| Bayesian aggregation top-k | 3 | 3 |

### Outputs

Each run writes to `experiments/<dataset>/<model>/<method>/<tasks>/<config>/<timestamp>_<strategy>/`, where `<method>` is `experts`, `utility_router`, `taskid_router`, or `naive`. The accuracy and calibration matrices are stored per evaluation scenario:

- `standard_*`: the router's top-1 expert
- `bayesian_*`: Bayesian aggregation over the top-k experts
- `oracle_*`: the expert of the ground-truth task

Summarize a run with:

```bash
python tools/summarize_vqa2.py -f <run directory>     # VQA v2
python tools/summarize_tdiuc.py -f <run directory>    # TDIUC
```

## Citation

```bibtex
@inproceedings{mozaffari2026calibrated,
  title     = {Calibrated Knowledge Aggregation in Bayesian Mixture-of-Experts for Continual {VQA}},
  author    = {Mozaffari, Mahsa and Sapkota, Hitesh and Kong, Yu and Liu, Xumin and Yu, Qi},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026}
}
```

## Acknowledgements

The data pipeline builds on [VQACL](https://github.com/zhangxi1997/VQACL), which in turn builds on [VL-T5](https://github.com/j-min/VL-T5). We thank the authors for releasing their code.

## License

This project is released under the [MIT License](LICENSE).
