# *coding:utf-8 *

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
from PIL import Image

from utils.image import randomHueSaturationValue
from utils.image import randomShiftScaleRotate
from utils.image import randomHorizontalFlip
from utils.image import randomVerticleFlip, randomRotate90


class BinarySegDataset(data.Dataset):

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.resize(img, (self.opt.height, self.opt.width))

        mask = np.array(Image.open(self.labels[index]))
        mask = cv2.resize(mask, (self.opt.height, self.opt.width))

        # Data augmentation
        if self.opt.color_aug:
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-30, 30),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))

        if self.opt.shift_scale:
            img, mask = randomShiftScaleRotate(img, mask,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.1, 0.1),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-0, 0))

        if self.opt.HorizontalFlip:
            img, mask = randomHorizontalFlip(img, mask)

        if self.opt.VerticleFlip:
            img, mask = randomVerticleFlip(img, mask)

        if self.opt.rotate_90:
            img, mask = randomRotate90(img, mask)

        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        ret = {'input': img, 'gt': mask}
        return ret
