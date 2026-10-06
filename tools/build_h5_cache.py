import h5py
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import os
from argparse import ArgumentParser
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class COCODataset(Dataset):
    def __init__(self, image_dir, img_size=384):
        self.image_dir = image_dir
        self.image_path_list = list(tqdm(image_dir.iterdir()))
        self.n_images = len(self.image_path_list)
        self.img_size = img_size

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_id = image_path.stem

        image = Image.open(image_path).convert('RGB')
        img_ = image.resize((self.img_size, self.img_size), Image.LANCZOS)

        return {
            'img_id': image_id,
            'img': img_
        }
    def collate_fn(self, batch):
        img_ids = []
        imgs = []

        for i, entry in enumerate(batch):
            img_ids.append(entry['img_id'])
            imgs.append(entry['img'])

        batch_out = {}
        batch_out['img_ids'] = img_ids
        batch_out['imgs'] = imgs

        return batch_out

def extract(output_fname, dataloader, desc):

    with h5py.File(output_fname, 'w') as f:
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader),
                                 desc=desc,
                                 ncols=150,
                                 total=len(dataloader)):

                img_ids = batch['img_ids']

                imgs = batch['imgs']

                assert len(imgs) == 1

                img = imgs[0]
                img_id = img_ids[0]

                try:
                    grp = f.create_group(img_id)
                    grp['img_w'] = img.size[1]
                    grp['img_h'] = img.size[0]
                    grp['image'] = img

                except Exception as e:
                    print(batch)
                    print(e)
                    continue

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--cocoroot', type=str, default='datasets/coco/images')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'])
    parser.add_argument('--outdir', type=str, default='coco_images.h5', help='Output H5 file path')
    parser.add_argument('--img_size', type=int, default=384, help='Image size (img_size x img_size)')

    args = parser.parse_args()

    SPLIT2DIR = {
        'train': 'train2014',
        'valid': 'val2014',
        'test': 'test2015',
    }

    coco_img_dir = Path(args.cocoroot).resolve()
    coco_img_split_dir = coco_img_dir.joinpath(SPLIT2DIR[args.split])

    dataset_name = 'COCO'

    out_dir = Path(args.outdir).resolve()
    if not out_dir.exists():
        out_dir.mkdir()

    print('Load images from', coco_img_split_dir)
    print('# Images:', len(list(coco_img_split_dir.iterdir())))

    dataset = COCODataset(coco_img_split_dir, img_size=args.img_size)

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=dataset.collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{args.split}_{args.img_size}.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{args.split}'

    extract(output_fname, dataloader, desc)
