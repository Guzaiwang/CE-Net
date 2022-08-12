# *coding:utf-8 *
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .binarySeg import BinarySegTrainer

train_factory = {
    'binSeg': BinarySegTrainer,
}