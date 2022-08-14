# *coding:utf-8 *
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.ORIGA_OD import ORIGA_OD
from .dataset.VOC import VOC
from .dataset.Cityscape import Cityscape
from .dataset.HumanSeg import HumanSeg

from .sample.binarySeg import BinarySegDataset
# from .sampel.multiSeg import MultiSegDataset

dataset_factory = {
    'ORIGA_OD': ORIGA_OD,
    'VOC': VOC,
    'CityScape': Cityscape,
    'humanseg': HumanSeg,
}

_sample_factory = {
    'binSeg': BinarySegDataset,
    # 'multiSeg': MultiSegDataset
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset
