# ------------------------------------------------------------------------
# INTR
# Copyright (c) 2023 PAUL. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
dataset which returns image and target for evaluation.
target contains the file name and image label of the image.
"""
import os
from pathlib import Path

import torch
import datasets.transforms as T
from torchvision.datasets import ImageFolder

from .constants import data_mean_std

class CreateImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        filename, imagelabel = self.samples[index]
        img = self.loader(filename)
        target={}
        target["file_name"]= [filename]
        target["image_label"]=torch.tensor([imagelabel], dtype=torch.int64) 
        
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

# The transform follows DETR transform 
def make_transforms(image_set, args):

    mean, std = data_mean_std["default"]

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):

    root = Path(args.dataset_path)
    assert root.exists(), f'provided dataset path {root} does not exist'
    data_file = Path(args.dataset_name)

    PATHS = {
        "train": (root / data_file / "train"),
        "val": (root / data_file / "val"),
        "test": (root / data_file / "test"),
    }

    img_folder = PATHS[image_set]

    if image_set == 'train':
        transform = make_transforms(image_set, args)
    elif image_set == 'val' or image_set == 'test':
        transform = make_transforms(image_set, args)
    else:
        raise ValueError(f'unknown {image_set}')

    dataset = CreateImageFolder(root=img_folder, transform=transform)

    return dataset





