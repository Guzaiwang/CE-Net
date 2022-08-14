# *coding:utf-8 *
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from _init_paths import __init_path

__init_path()

import os

import torch
import torch.utils

from logger import Logger
from datasets.dataset_factory import get_dataset
from models.model import create_model, load_model, save_model
from trains.train_factory import train_factory

from opts import opts


def main(opt):
    # Completely reproducible results are not guaranteed across PyTorch releases, \
    # individual commits, or different platforms. Furthermore, results may not be reproducible \
    # between CPU and GPU executions, even when using identical seeds.
    # We can use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):
    torch.manual_seed(opt.seed)

    # 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，\
    # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。\
    # 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，\
    # 其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    Dataset = get_dataset(opt.dataset, opt.task)

    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')

    model = create_model(opt.model_name)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    Trainer = train_factory[opt.task]

    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print("Setting up data...")
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        # run evaluation code
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print("Starting training")
    best = 1e10

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))

        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)

        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
