# ------------------------------------------------------------------------
# INTR
# Copyright (c) 2023 PAUL. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch
import util.misc as utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):

    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)

        filenames=[t["file_name"] for t in targets]
        for t in targets:
            del t["file_name"]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs, _ ,_ ,_ ,_  = model(samples)
        loss_dict = criterion(outputs, targets, model)

        ## INTR uses only one type of loss i.e., CE loss
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        ## reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value =sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        acc1, acc5, _ = utils.class_accuracy(outputs, targets, topk=(1, 5))

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1)
        metric_logger.update(acc5=acc5)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion,  data_loader, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        filenames=[t["file_name"] for t in targets]
        for t in targets:
            del t["file_name"]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs,_,_,_,_ = model(samples)
        loss_dict = criterion(outputs, targets, model)

        ## reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value =sum(loss_dict_reduced.values())

        metric_logger.update(loss=loss_value)                                   
        acc1, acc5, _ = utils.class_accuracy(outputs, targets, topk=(1, 5))
        metric_logger.update(acc1=acc1)
        metric_logger.update(acc5=acc5)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats