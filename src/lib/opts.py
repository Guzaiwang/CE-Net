# *coding:utf-8 *
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiments setting
        self.parser.add_argument('--task', default='binSeg',
                                 help='binSeg'
                                      'binSeg: Binary Segmentation')
        self.parser.add_argument('--dataset', default='ORIGA_OD',
                                 help='ORIGA_OD | kitti | coco_hp | pascal')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--debug', type=int, default=0,
                                 help='level of visualization.'
                                      '1: only show the final detection results'
                                      '2: show the network output features'
                                      '3: use matplot to display'  # useful when lunching training with ipython notebook
                                      '4: save all visualizations to disk')
        self.parser.add_argument('--demo', default='',
                                 help='path to image/ image folders/ video. '
                                      'or "webcam"')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # model
        self.parser.add_argument('--model_name', default='cenet',
                                 help='which model')

        # data_root
        self.parser.add_argument('--data_root', default='/data/zaiwang/Dataset',
                                 help='which model')

        # loss type
        self.parser.add_argument('--loss_type', default='dice_bce_loss',
                                 help='which loss type')

        # system
        self.parser.add_argument('--gpus', default='3',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                                 help='visualization threshold.')
        self.parser.add_argument('--debugger_theme', default='white',
                                 choices=['white', 'black'])

        # input
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='90,120',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=140,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=24,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')

        self.parser.add_argument('--root_dir', type=str, default='/data/zaiwang/output',
                                 help='the path to save training model and loggers')

        # dataset augmentation
        self.parser.add_argument('--color_aug', default=True,
                                 help='HSV color augmentation')

        self.parser.add_argument('--shift_scale', default=True,
                                 help='when not using random crop'
                                      'apply shift augmentation.')

        self.parser.add_argument('--HorizontalFlip', default=True,
                                 help='Horizontal Flip')

        self.parser.add_argument('--VerticleFlip', default=True,
                                 help='Verticle Flip')

        self.parser.add_argument('--rotate_90', default=True,
                                 help='when not using random crop'
                                      'apply rotation augmentation.')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        print("GPUS device is {}".format(opt.gpus))

        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        if opt.debug > 0:
            opt.num_workers = 0
            opt.batch_size = 1
            opt.gpus = [opt.gpus[0]]
            opt.master_batch_size = -1

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.data_dir = os.path.join(opt.data_root, opt.dataset)
        opt.exp_dir = os.path.join(opt.root_dir, 'UBT_Seg', opt.task)
        opt.exp_id = opt.dataset + '_' + opt.model_name + '_' + opt.loss_type
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        opt.height, opt.width = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        return opt

    def init(self, args=''):
        default_dataset_info = {
            'binSeg': {'default_resolution': [512, 512], 'num_classes': 1,
                       'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                       'dataset': 'ORIGA_OD'},

        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
