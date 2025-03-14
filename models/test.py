#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import DatasetSplit


def test_img(net_g, datatest, args, idx=None):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    if idx is not None:
        data_loader = DataLoader(DatasetSplit(datatest, idx), batch_size=args.local_bs)
    else:
        data_loader = DataLoader(datatest, batch_size=args.bs)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(args.device), target.cuda(args.device)
            log_probs = net_g(data)['output']
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

