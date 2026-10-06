
from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
import re
import os
from PIL import Image
import logging

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer

import sys
sys.path.append("..")
from torch.utils.data import random_split
from torch.utils.data import DataLoader #, ConcatDataset

class VQAViltFineTuneDataset(Dataset):
    def __init__(self, processor, image_dir, coco_Ours, Examplar_set, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train', task='q_what', cates=[0,1,2], task_id=None, ans2label=None, vqa_dir=None, partition_name='Partition_Q', use_h5=False, h5_path=None):
        super().__init__()

        self.logger = logging.getLogger(f'CL.data.vqa_dataset.{task}')
        self.vqa_dir = Path(vqa_dir) if vqa_dir else globals().get('vqa_dir')

        self.processor = processor

        self.image_dir = image_dir
        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode
        self.task_id = task_id

        self.partition_name = partition_name

        self.check_im_cate = False

        # Loading datasets to data
        self.sources = split.split(',')

        if ans2label is None:
        # Topk Answers
            self.ans2label = json.load(
                open(self.vqa_dir.joinpath(f"{self.partition_name}/{task}_ans2label.json")))
        else:
            self.ans2label = ans2label
        #sort based on values
        self.label2ans = [k for k, v in sorted(self.ans2label.items(), key=lambda item: item[1])]
        for idx, ans in enumerate(self.label2ans):
            assert self.ans2label[ans] == idx, \
                f"Inconsistency during initialization: label2ans[{idx}]={ans} but ans2label[{ans}]={self.ans2label[ans]}"

        assert len(self.ans2label) == len(self.label2ans)
        self.num_answers = len(self.ans2label)

        # Log the dataset state after answer re-indexing.
        self.logger.info(f"=== DATASET INITIALIZED for task {task} ===")
        self.logger.info(f"  Original ans2label size: {len(self.ans2label)}")

        # Verify consistency
        for idx, ans in enumerate(self.label2ans):
            assert self.ans2label[ans] == idx, \
                f"Inconsistency: label2ans[{idx}]={ans} but ans2label[{ans}]={self.ans2label[ans]}"

        self.answer_normalizer = VQAEvaluator()

        self.img_ids_to_source = {}
        data_info_dicts_cate = []
        self.cate_set = set()
        for source in self.sources:
            data_info_path = self.vqa_dir.joinpath(f'{self.partition_name}/{source}_'+f'{task}.json')

            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                _data_info_dicts.extend(Examplar_set)
                for _d in _data_info_dicts:
                    img_id = _d['img_id']
                    try:
                        data_info_dicts_cate.append(_d)
                        if 'vg_qa_full' == source:
                            self.img_ids_to_source[_d['img_id']] = 'vg'
                        elif 'train2014' in _d['img_id']:
                            self.img_ids_to_source[_d['img_id']] = 'train2014'
                        elif 'val2014' in _d['img_id']:
                            self.img_ids_to_source[_d['img_id']] = 'val2014'
                        else:
                            self.img_ids_to_source[_d['img_id']] = source
                            _d['source'] = source
                    except:
                        continue

        self.use_h5 = use_h5
        self.h5_path = h5_path
        if use_h5:
            if h5_path is None:
                raise ValueError("h5_path must be provided when use_h5 is True")
            if not os.path.exists(h5_path):
                raise ValueError(f"h5_path {h5_path} does not exist")
        self.source_to_h5 = {
            'train': None,
            'minival': None,
            'nominival': None,
            'test': os.path.join(h5_path, f'test_384.h5'),
            'vg': None,
            'train2014': os.path.join(h5_path, (f'train_384.h5')),
            'val2014': os.path.join(h5_path, (f'valid_384.h5')),
        }

        data = data_info_dicts_cate

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                self.logger.info(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            self.logger.info(f"# all sentences: {len(self.data)} with Examplers")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        datum = self.data[idx]

        img_id = datum['img_id']
        source = self.img_ids_to_source[img_id] # source: val2014

        if self.use_h5:
            f = self.source_to_h5[source]
            if isinstance(f, str) and os.path.exists(f):
                f = h5py.File(f, 'r', swmr=True)
                self.source_to_h5[source] = f
            try:
                image = f[img_id]['image'][()]
                image = Image.fromarray(image).convert('RGB')
            except:
                image = None

        if not self.use_h5 or image is None:
            image_path = os.path.join(self.image_dir, source, f"{img_id}.jpg")
            image = Image.open(image_path).convert('RGB')

        img_ = image

        # Get question
        question = datum['question'] if 'question' in datum else datum['sent']

        # Process image and text using ViLT processor

        item = {
            'question_id': datum['question_id'],
            'img_id': img_id,
            'image': img_,
            'sent': question,

        }
        # Remove batch dimension added by processor

        # Add additional information

        if 'label' in datum:
            label = datum['label']
            item['label'] = label

            # 3129 topk answers
            if self.args.classifier:
                target = torch.zeros(self.num_answers)
                for ans, score in label.items():
                    if ans in self.ans2label:       # Added this because of TDIUC dataset
                        target[self.ans2label[ans]] = score
                item['target'] = target

            elif self.args.raw_label:
                raise NotImplementedError

                # 10 raw answers
                # ex) 'answers': [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 2},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 3},
                #     {'answer': 'netting', 'answer_confidence': 'yes', 'answer_id': 4},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 5},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 6},
                #     {'answer': 'mesh', 'answer_confidence': 'maybe', 'answer_id': 7},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 8},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 9},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 10}],

                answers = datum['answers']
                answer = random.choice(answers)['answer']

                if self.args.answer_normalize:
                    answer = self.answer_normalizer.normalize_answer(answer)

                score = int(len(answers) > 0)

                out_dict['answer'] = answer
                out_dict['score'] = score
                out_dict['all_answers'] = [a['answer'] for a in answers]

                target_ids = self.tokenizer.encode(answer, max_length=10, truncation=True)

                out_dict['target_ids'] = torch.LongTensor(target_ids)
                out_dict['target_length'] = len(target_ids)

            else:
                # https://github.com/airsplay/lxmert/blob/master/src/pretrain/lxmert_pretrain.py#L191
                answers = []
                scores = []
                for a, s in label.items():
                    answers.append(a)
                    scores.append(s)

                score_sum = sum(scores)

                if score_sum == 0:
                    answer = ''
                    score = 0.
                else:
                    prob = [score / score_sum for score in scores]
                    choice = np.random.multinomial(1, prob).argmax()
                    answer = answers[choice]
                    score = scores[choice]
                    assert len(answer) > 0, (sent, label, choice, answer)

                item['answer'] = answer
                item['score'] = score
                item['all_answers'] = answers

        if self.task_id is not None:
            item['task_id'] = self.task_id

        return item

    def __del__(self):
        # Close H5 files when dataset is destroyed
        for f in self.source_to_h5.values():
            if f and isinstance(f, h5py.File):
                f.close()

    def collate_fn(self, batch):
        batch_dict = {}

        keys = batch[0].keys()

        images = [t['image'] for t in batch]
        questions = [t['sent'] for t in batch]

        encoding = self.processor(
            images=images,
            text=questions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=40,
        )
        # Keep sequence length fixed for compatibility with the current model setup.

        # Start with the processed encoding
        batch_dict = dict(encoding)

        B = len(batch)

        for key in keys:

            if key in ['image']:# , 'sent']:
                continue

            if key == 'target':
                batch_dict['targets'] = torch.stack([item[key] for item in batch])
            elif key in ['task_id']:
                values = torch.LongTensor([item[key] for item in batch]).unsqueeze(1)
                if key == 'task_id':
                    batch_dict['task_ids'] = values

            elif key == 'score' and 'score' in batch[0]:
                batch_dict['scores'] = torch.tensor([item.get(key, 0.0) for item in batch])

            # Keep label, answer, all_answers as lists (metadata)
            elif key in ['label', 'answer', 'all_answers']:
                # Store as list - this won't cause issues with pin_memory
                batch_dict[key] = [item.get(key) for item in batch]

            elif key in ['sent']:
                batch_dict['questions'] = [item[key] for item in batch]

            # Keep these as metadata (won't be moved to GPU anyway)
            elif key in ['question_id', 'img_id']:
                batch_dict[key] = [item[key] for item in batch]

            # Skip non-essential fields that can't be tensorized
            else:
                batch_dict[key] = [item[key] for item in batch]

        return batch_dict

def get_loader_test(processor, image_dir, args, coco_Ours, Examplar_set, _dset, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1, task='what', task_id = None, ans2label=None):

    verbose = (gpu == 0)

    dataset = VQAViltFineTuneDataset(
        processor,
        image_dir,
        coco_Ours,
        Examplar_set,
        split,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode,
        task=task,
        task_id=task_id,
        ans2label=ans2label) # all categories

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = VQAEvaluator(_dset)

    loader.task = 'vqa'
    return loader

def get_loader(processor, image_dir, args, coco_Ours, Examplar_set, _dset, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1, task='what', re_split=False, split_ratio=0.8, task_id = None, ans2label=None):

    verbose = (gpu == 0)

    total_num = 0
    cate_loader = {}

    for idx, CateGroup in enumerate(Category_splits):
        print(CateGroup, end=',')
        dataset=VQAViltFineTuneDataset(
            processor,
            image_dir,
            coco_Ours,
            Examplar_set,
            split,
            raw_dataset=_dset,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            mode=mode,
            task=task,
            task_id=task_id,    ## ?
            ans2label=ans2label ## ?
        )
        total_num += len(dataset)

        if distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = None

        if mode == 'train':
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=(sampler is None),
                num_workers=workers, pin_memory=True, sampler=sampler,
                collate_fn=dataset.collate_fn)
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=workers, pin_memory=True,
                sampler=sampler,
                shuffle=None if (sampler is not None) else False,
                collate_fn=dataset.collate_fn,
                drop_last=False)

        if verbose:
            loader.evaluator = VQAEvaluator(_dset)

        loader.task = 'vqa'

        cate_loader[CateGroup] = loader

    return cate_loader, total_num


def get_loader_qlevel(processor, image_dir, args, coco_Ours, Examplar_set, _dset, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1, task='what', re_split=False, split_ratio=0.8, task_id = None, ans2label=None, partition_name='Partition_Q',
               persistent_workers=False, prefetch_factor=2, pin_memory=True):

    verbose = (gpu == 0)

    total_num = 0

    dataset = VQAViltFineTuneDataset(
        processor,
        image_dir,
        coco_Ours,
        Examplar_set,
        split,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode,
        task=task,
        task_id=task_id,
        ans2label=ans2label,
        vqa_dir=args.vqa_dir,
        partition_name=partition_name,
        use_h5=getattr(args, 'use_h5', False),
        h5_path=getattr(args, 'h5_path', None))
    total_num = len(dataset)

    if re_split:
        dev_size = int(split_ratio * len(dataset))
        rest_size = total_num - dev_size
        main_dataset, rest_dataset = random_split(dataset.data, [dev_size, rest_size])
        dataset.data = main_dataset
        import copy
        rest = copy.deepcopy(dataset)
        rest.data = rest_dataset
    else:
        rest = None

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if rest is not None:
        rest_sampler = DistributedSampler(rest) if distributed else None
        rest_loader = DataLoader(
            rest, batch_size=batch_size, shuffle=(rest_sampler is None),
            num_workers=workers, pin_memory=pin_memory, sampler=rest_sampler,
            collate_fn=dataset.collate_fn, persistent_workers=persistent_workers and workers > 0,
            prefetch_factor=prefetch_factor if workers > 0 else None, timeout=300, drop_last=False)
    else:
        rest_loader = None

    if mode == 'train':
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            persistent_workers=persistent_workers and workers > 0,
            prefetch_factor=prefetch_factor if workers > 0 else None,
            multiprocessing_context='fork' if workers > 0 else None,  # Faster than spawn
            timeout=300, drop_last=False)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=pin_memory,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False, persistent_workers=persistent_workers and workers > 0,
            prefetch_factor=prefetch_factor if workers > 0 else None, timeout=300)

    if verbose:
        loader.evaluator = VQAEvaluator(_dset)
        if rest_loader is not None:
            rest_loader.evaluator = VQAEvaluator(_dset)

    loader.task = 'vqa'

    return loader, total_num , dataset.num_answers, dataset.label2ans, rest_loader

class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """

    def __init__(self, splits: str, verbose=True, vqa_dir = None):

        self.vqa_dir = Path(vqa_dir) if vqa_dir else globals().get("vqa_dir")

        self.name = splits
        self.splits = splits.split(',')

        with open(self.vqa_dir.joinpath(f'v2_mscoco_train2014_annotations.json')) as f:
            train2014_data = json.load(f)

        with open(self.vqa_dir.joinpath(f'v2_mscoco_val2014_annotations.json')) as f:
            val2014_data = json.load(f)

        train2014_id2datum = {}
        for datum in train2014_data['annotations']:
            qid = datum['question_id']
            train2014_id2datum[qid] = datum
        val2014_id2datum = {}
        for datum in val2014_data['annotations']:
            qid = datum['question_id']
            val2014_id2datum[qid] = datum
        self.id2datum_gt = {**train2014_id2datum, **val2014_id2datum}

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(
                json.load(open(self.vqa_dir.joinpath(f"{split}.json"))))

        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Topk Answers
        self.ans2label = json.load(
            open(self.vqa_dir.joinpath("trainval_ans2label.json")))
        self.label2ans = json.load(
            open(self.vqa_dir.joinpath("trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)

        if verbose:
            print('# All Answers:', len(self.ans2label))

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

class TDIUCDataset(VQADataset):
    def __init__(self, splits: str, verbose=True, vqa_dir=None):
        self.vqa_dir = Path(vqa_dir) if vqa_dir else dataset_dir.joinpath('tdiuc')

        self.name = splits
        self.splits = splits.split(',')

        with open(self.vqa_dir.joinpath('mscoco_train2014_annotations.json')) as f:
            train2014_data = json.load(f)

        with open(self.vqa_dir.joinpath('mscoco_val2014_annotations.json')) as f:
            val2014_data = json.load(f)

        train2014_id2datum = {}
        for datum in train2014_data['annotations']:
            qid = datum['question_id']
            train2014_id2datum[qid] = datum
        val2014_id2datum = {}
        for datum in val2014_data['annotations']:
            qid = datum['question_id']
            val2014_id2datum[qid] = datum

        self.id2datum_gt = {**train2014_id2datum, **val2014_id2datum}
        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(
                json.load(open(self.vqa_dir.joinpath(f"{split}.json"))))
        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
        # Topk Answers
        self.ans2label = json.load(
            open(self.vqa_dir.joinpath("train_ans2label.json")))
        self.label2ans = json.load(
            open(self.vqa_dir.joinpath("train_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)
        if verbose:
            print('# All Answers:', len(self.ans2label))

class VQAEvaluator:
    def __init__(self, dataset: VQADataset = None):
        self.dataset = dataset
        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}

        self.manualMap    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}

        self.articles     = ['a',
							 'an',
							 'the'
							]

        self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile("(\d)(\,)(\d)")
        self.punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '\\', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']

        self.n = 2

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            if len(ans) > 1:
                ans = ans['answer']
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }
        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

    def evaluate_raw(self, quesid2ans: dict, is_topk_optimal=None):
        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        gts = self.dataset.id2datum_gt

        self.accuracy     = {}
        self.evalQA       = {}
        self.evalQuesType = {}
        self.evalAnsType  = {}

        accQA = []
        accQuesType = {}
        accAnsType = {}

        for quesId, resItem in tqdm(quesid2ans.items(), total=len(quesid2ans), ncols=80):

            resAns = resItem['answer']
            quesId = int(quesId)

            datum = self.dataset.id2datum[quesId]

            if is_topk_optimal is None:
                pass
            elif 'is_topk_optimal' in datum:
                if datum['is_topk_optimal'] != is_topk_optimal:
                    continue

            resAns      = resAns.replace('\n', ' ')
            resAns      = resAns.replace('\t', ' ')
            resAns      = resAns.strip()
            resAns      = self.processPunctuation(resAns)
            resAns      = self.processDigitArticle(resAns)

            gtAcc  = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]['answers']:
                    ansDic['answer'] = self.processPunctuation(ansDic['answer'])
            for gtAnsDatum in gts[quesId]['answers']:
                otherGTAns = [item for item in gts[quesId]['answers'] if item!=gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item['answer']==resAns]
                acc = min(1, float(len(matchingAns))/3)
                gtAcc.append(acc)

            quesType    = gts[quesId]['question_type']
            ansType     = gts[quesId]['answer_type']
            avgGTAcc = float(sum(gtAcc))/len(gtAcc)
            accQA.append(avgGTAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)

            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)

        if len(accQA) == 0:
            return {
                'overall': 0,
                'perQuestionType': {},
                'perAnswerType': {}
            }
        else:
            self.setAccuracy(accQA, accQuesType, accAnsType)

        return self.accuracy

    def evaluate_raw_ece(self, quesid2ans: dict, is_topk_optimal=None):
        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        gts = self.dataset.id2datum_gt

        self.accuracy     = {}
        self.mc_accuracy = {}
        self.evalQA       = {}
        self.evalQuesType = {}
        self.evalAnsType  = {}

        accQA = []
        MCaccQA = []
        confs = []
        accQuesType = {}
        accAnsType = {}

        def get_mc_gt_answer(labels):
            max_lbl = None
            max_score = float('-inf')
            for label, score in labels.items():
                if score > max_score:
                    max_score = score
                    max_lbl = label
            if max_lbl is None:
                return None
            max_lbl = self.processPunctuation(max_lbl)
            max_lbl = self.processDigitArticle(max_lbl)
            return max_lbl

        for quesId, resItem in tqdm(quesid2ans.items(), total=len(quesid2ans), ncols=80):

            quesId = int(quesId)

            resConf = resItem['confidence']
            resAns = resItem['answer']

            datum = self.dataset.id2datum[quesId]
            mc_gt_answer = get_mc_gt_answer(datum['label'])

            if is_topk_optimal is None:
                pass
            elif 'is_topk_optimal' in datum:
                if datum['is_topk_optimal'] != is_topk_optimal:
                    continue

            resAns      = resAns.replace('\n', ' ')
            resAns      = resAns.replace('\t', ' ')
            resAns      = resAns.strip()
            resAns      = self.processPunctuation(resAns)
            resAns      = self.processDigitArticle(resAns)

            for ansDic in gts[quesId]['answers']:
                ansDic["answer"] = self.processPunctuation(ansDic["answer"])
                ansDic["answer"] = self.processDigitArticle(ansDic["answer"])

            gtAcc  = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
            if len(set(gtAnswers)) > 1:
                for gtAnsDatum in gts[quesId]['answers']:
                    otherGTAns = [item for item in gts[quesId]['answers'] if item!=gtAnsDatum]
                    matchingAns = [item for item in otherGTAns if item['answer']==resAns]
                    acc = min(1, float(len(matchingAns))/3)
                    gtAcc.append(acc)
            else:
                gt_item = gts[quesId]["answers"][0]
                acc = float(1) if resAns == gt_item["answer"] else float(0)
                gtAcc.append(acc)

            quesType    = gts[quesId]['question_type']
            ansType     = gts[quesId]['answer_type']
            avgGTAcc = float(sum(gtAcc))/len(gtAcc)
            accQA.append(avgGTAcc)
            confs.append(resConf)
            MCaccQA.append(1 if mc_gt_answer == resAns else 0)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)

            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)

        if len(accQA) == 0:
            return {
                'overall': 0,
                'perQuestionType': {},
                'perAnswerType': {}
            }
        else:
            self.setAccuracy(accQA, accQuesType, accAnsType)
            self.setEvalECE(MCaccQA, confs)

        return self.accuracy

    def setEvalECE(self, acc, conf):
        ece = self._calc_ece_result(acc, conf)
        self.accuracy['ece'] = ece

    def _calc_ece_result(self, accuracies, confidences):
        bin_boundaries = torch.linspace(0, 1, 10 + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies = torch.Tensor(accuracies)
        confidences = torch.Tensor(confidences)
        ece = torch.zeros(1)
        ece_x_axis = []
        ece_y_axis = []
        ece_y_axis_std = []
        ece_x_axis_cnt = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                ece_x_axis_cnt.append(prop_in_bin.item())

                std, accuracy_in_bin = torch.std_mean(accuracies[in_bin].float())
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_y_axis_std.append(std.item())
                ece_x_axis.append(avg_confidence_in_bin.item())
                ece_y_axis.append(accuracy_in_bin.item())
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        ece = ece.item()
        return ece

    def normalize_answer(self, resAns):
        resAns      = resAns.replace('\n', ' ')
        resAns      = resAns.replace('\t', ' ')
        resAns      = resAns.strip()
        resAns      = self.processPunctuation(resAns)
        resAns      = self.processDigitArticle(resAns)
        resAns = resAns.replace(',', '')
        return resAns

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                        outText,
                                        re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100*acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100*acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100*acc, self.n)

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy['overall']         = round(100*float(sum(accQA))/len(accQA), self.n)
        self.accuracy['perQuestionType'] = {quesType: round(100*float(sum(accQuesType[quesType]))/len(accQuesType[quesType]), self.n) for quesType in accQuesType}
        self.accuracy['perAnswerType']   = {ansType:  round(100*float(sum(accAnsType[ansType]))/len(accAnsType[ansType]), self.n) for ansType in accAnsType}
