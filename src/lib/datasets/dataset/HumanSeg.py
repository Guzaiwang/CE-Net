# *coding:utf-8 *

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os

import torch.utils.data as data


class HumanSeg(data.Dataset):
    num_classes = 1
    default_resolution = [448, 448]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    def __init__(self, opt, split):
        self.images = []
        self.labels = []
        self.opt = opt

        if split == 'train':
            self.image_root_folder = os.path.join(opt.data_dir,'test', 'imgs')
            self.gt_root_folder = os.path.join(opt.data_dir,'test', 'masks')
        else:
            self.image_root_folder = os.path.join(opt.data_dir, 'val', 'imgs')
            self.gt_root_folder = os.path.join(opt.data_dir, 'val', 'masks')

        self._read_img_mask(self.image_root_folder, self.gt_root_folder)

    def _read_img_mask(self, image_folder, mask_folder):
        for img_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, img_name.split('.')[0] + '.png')
            label_path = os.path.join(mask_folder, img_name.split('.')[0] + '.png')

            self.images.append(image_path)
            self.labels.append(label_path)

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)