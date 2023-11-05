# ------------------------------------------------------------------------
# INTR
# Copyright (c) 2023 Imageomics Paul. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import json
import time
import random
import datetime
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
from pathlib import Path

import datasets
import util.misc as utils
from models import build_model
from datasets import build_dataset
from engine import evaluate, train_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=5.00e-5, type=float) 
    parser.add_argument('--lr_backbone', default=1.00e-5, type=float) 
    parser.add_argument('--min_lr', default=1.00e-6, type=float) 
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--epochs', default=140, type=int) 
    parser.add_argument('--lr_drop', default=80, type=int) 
    parser.add_argument('--lr_scheduler', default="StepLR", type=str, choices=["StepLR", "CosineAnnealingLR"])
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Backbone parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer parameters
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer") 
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer") 
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=200, type=int,
                        help="Number of query slots which equals to number of classes")
    parser.add_argument('--pre_norm', action='store_true')


    # * Dataset parameters
    parser.add_argument('--dataset_name', default='cub', type=str) 
    parser.add_argument('--dataset_path', default='/path/to/datasets', type=str) 
    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_sub_dir', default='output_sub')
    
    # * INTR parameters
    parser.add_argument('--noise_frac', default=0.1, type=float,
                        help='fraction of noise to be added to new queries while loading pretrained model')
    # parser.add_argument('--rm_freeze', default=140, type=int, help='epoch at which the freezing at the encoder is removed')
    parser.add_argument('--test', default="val", type=str, choices=["val", "test"])
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--finetune', default='', 
                        help='finetune from pretrained checkpoint (COCO dataset trained for object detection task)')

    # * Device parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')


    # * Distributed training parameters
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion= build_model(args)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        ## for 2-phase training
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True) 
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    if args.lr_scheduler=="StepLR":
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.lr_scheduler=="CosineAnnealingLR":
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, 
                                    weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set=args.test, args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    #   We create output directories to store results
    output_dir = Path(args.output_dir)
    if not os.path.exists(os.path.join(output_dir, args.dataset_name)):
        os.makedirs(os.path.join(output_dir, args.dataset_name), exist_ok=True)
    if not os.path.exists(os.path.join(output_dir, args.dataset_name, args.output_sub_dir)):
        os.makedirs(os.path.join(output_dir, args.dataset_name, args.output_sub_dir), exist_ok=True)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats = evaluate(model, criterion, 
                                data_loader_val, device, args.output_dir)
        if args.output_dir and utils.is_main_process():
            with (output_dir / args.dataset_name / args.output_sub_dir/ "log.txt").open("a") as f:
                f.write(json.dumps(test_stats) + "\n")
        return

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        state_dict = checkpoint['model']
        state_dict=utils.load_model(args, state_dict)
        
        model_without_ddp.load_state_dict(state_dict)

        for param in model_without_ddp.parameters():
            param.requires_grad = True
        model_without_ddp.to(device)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        ## for 2-phase training
        # if epoch>=args.rm_freeze:
        #     for param in model_without_ddp.transformer.encoder.parameters():
        #         param.requires_grad = True

        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / args.dataset_name / args.output_sub_dir/ 'checkpoint.pth']

            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1)==args.epochs:
                checkpoint_paths.append(output_dir / args.dataset_name / args.output_sub_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(
            model, criterion,  data_loader_val, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / args.dataset_name / args.output_sub_dir/ "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('INTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)