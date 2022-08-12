# *coding:utf-8 *

import torch
import numpy as np

from models.losses import dice_bce_loss
from .base_trainer import BaseTrainer

class binarySegLoss(torch.nn.Module):
    def __init__(self, opt):
        super(binarySegLoss, self).__init__()
        self.crit = dice_bce_loss()
        self.opt = opt

    def forward(self, outputs, batch):

        loss = self.crit(batch['gt'], outputs)

        loss_stats = {'loss': loss}

        return loss, loss_stats


class BinarySegTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(BinarySegTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_stats = ['loss']
        loss = binarySegLoss(opt)
        return loss_stats, loss

