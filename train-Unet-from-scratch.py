# Copyright 2019 the UBtech intern Zaiwang.Gu. All Rights Resvered.
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
import numpy as np

from time import time

from visdom import Visdom
from Visualizer import Visualizer
from networks.UNet import UNet
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder


def vessel_main():
    SHAPE = (448, 448)
    # ROOT = 'dataset/RIM-ONE/'
    ROOT = './dataset/DRIVE'
    NAME = 'log01_dink34-UNet' + ROOT.split('/')[-1]
    BATCHSIZE_PER_CARD = 8

    # net = UNet(n_channels=3, n_classes=2)

    viz = Visualizer(env="Vessel_Unet_from_scratch")

    solver = MyFrame(UNet, dice_bce_loss, 2e-4)
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

    dataset = ImageFolder(root_path=ROOT, datasets='DRIVE')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)

    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()
    no_optim = 0
    total_epoch = 300
    train_epoch_best_loss = 10000.
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0

        index = 0

        for img, mask in data_loader_iter:
            solver.set_input(img, mask)

            train_loss, pred = solver.optimize()

            train_epoch_loss += train_loss

            index = index + 1

            # if index % 10 == 0:
            #     # train_epoch_loss /= index
            #     # viz.plot(name='loss', y=train_epoch_loss)
            #     show_image = (img + 1.6) / 3.2 * 255.
            #     viz.img(name='images', img_=show_image[0, :, :, :])
            #     viz.img(name='labels', img_=mask[0, :, :, :])
            #     viz.img(name='prediction', img_=pred[0, :, :, :])

        show_image = (img + 1.6) / 3.2 * 255.
        viz.img(name='images', img_=show_image[0, :, :, :])
        viz.img(name='labels', img_=mask[0, :, :, :])
        viz.img(name='prediction', img_=pred[0, :, :, :])

        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        print(mylog, '********')
        print(mylog, 'epoch:', epoch, '    time:', int(time() - tic))
        print(mylog, 'train_loss:', train_epoch_loss)
        print(mylog, 'SHAPE:', SHAPE)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', SHAPE)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/' + NAME + '.th')
        if no_optim > 20:
            print(mylog, 'early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 15:
            if solver.old_lr < 5e-7:
                break
            solver.load('./weights/' + NAME + '.th')
            solver.update_lr(2.0, factor=True, mylog=mylog)
        mylog.flush()

    print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()


if __name__ == '__main__':

    # print(torch.cuda.device_count())
    vessel_main()
