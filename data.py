"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image

import cv2
import numpy as np
import os
import scipy.misc as misc

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask


def default_loader(img_path, mask_path):

    img = cv2.imread(img_path)
    # print("img:{}".format(np.shape(img)))
    img = cv2.resize(img, (448, 448))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = 255. - cv2.resize(mask, (448, 448))
    
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    
    mask = np.expand_dims(mask, axis=2)
    #
    # print(np.shape(img))
    # print(np.shape(mask))

    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    #mask = abs(mask-1)
    return img, mask


def default_DRIVE_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (448, 448))
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.array(Image.open(mask_path))

    mask = cv2.resize(mask, (448, 448))

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask

def read_ORIGA_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'Set_A.txt')
    else:
        read_files = os.path.join(root_path, 'Set_B.txt')

    image_root = os.path.join(root_path, 'images')
    gt_root = os.path.join(root_path, 'masks')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.jpg')

        print(image_path, label_path)

        images.append(image_path)
        masks.append(label_path)

    return images, masks

def read_Messidor_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'train.txt')
    else:
        read_files = os.path.join(root_path, 'test.txt')

    image_root = os.path.join(root_path, 'save_image')
    gt_root = os.path.join(root_path, 'save_mask')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.png')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.png')

        images.append(image_path)
        masks.append(label_path)

    return images, masks

def read_RIM_ONE_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'train_files.txt')
    else:
        read_files = os.path.join(root_path, 'test_files.txt')

    image_root = os.path.join(root_path, 'RIM-ONE-images')
    gt_root = os.path.join(root_path, 'RIM-ONE-exp1')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.png')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '-exp1.png')

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def read_DRIVE_datasets(root_path, mode='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, 'training/images')
    gt_root = os.path.join(root_path, 'training/1st_manual')


    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
        label_path = os.path.join(gt_root, image_name.split('_')[0] + '_manual1.gif')

        images.append(image_path)
        masks.append(label_path)

    print(images, masks)

    return images, masks


def read_Cell_datasets(root_path, mode='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, 'train-images')
    gt_root = os.path.join(root_path, 'train-labels')


    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)

        images.append(image_path)
        masks.append(label_path)

    print(images, masks)

    return images, masks


def read_datasets_vessel(root_path, mode='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, 'training/images')
    gt_root = os.path.join(root_path, 'training/mask')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)

        if cv2.imread(image_path) is not None:

            if os.path.exists(image_path) and os.path.exists(label_path):

                images.append(image_path)
                masks.append(label_path)

    print(images[:10], masks[:10])

    return images, masks


class ImageFolder(data.Dataset):

    def __init__(self,root_path, datasets='Messidor',  mode='train'):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        assert self.dataset in ['RIM-ONE', 'Messidor', 'ORIGA', 'DRIVE', 'Cell', 'Vessel'], \
            "the dataset should be in 'Messidor', 'ORIGA', 'RIM-ONE', 'Vessel' "
        if self.dataset == 'RIM-ONE':
            self.images, self.labels = read_RIM_ONE_datasets(self.root, self.mode)
        elif self.dataset == 'Messidor':
            self.images, self.labels = read_Messidor_datasets(self.root, self.mode)
        elif self.dataset == 'ORIGA':
            self.images, self.labels = read_ORIGA_datasets(self.root, self.mode)
        elif self.dataset == 'DRIVE':
            self.images, self.labels = read_DRIVE_datasets(self.root, self.mode)
        elif self.dataset == 'Cell':
            self.images, self.labels = read_Cell_datasets(self.root, self.mode)
        elif self.dataset == 'GAN_Vessel':
            self.images, self.labels = read_datasets_vessel(self.root, self.mode)
        else:
            print('Default dataset is Messidor')
            self.images, self.labels = read_Messidor_datasets(self.root, self.mode)

    def __getitem__(self, index):

        img, mask = default_DRIVE_loader(self.images[index], self.labels[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)