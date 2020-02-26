# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
# 对数据进行采样的
from torch.utils.data.sampler import Sampler


# ×××下面这句代码，在import的时候，会执行到/lib/model/utils/config.py文件，将配置文件中的内容都读取到变量cfg中。
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet


def parse_args():
    """
    Parse input arguments
    """
    # dest参数用来指定参数的位置
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    # 这个输入True还是False
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    # 是否执行类无关的bbox回归
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    # 判断数据库
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        # ×××需要注意的是的，这里使用2007的test作为验证集
        args.imdbval_name = "voc_2007_test"
        # 设置一些必要的参数，anchor的大小和宽高比
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    # 0712是用07和12训练，07测试集进行验证
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    # 这里的这个args.net指的是 特征提取层的模型，是使用res还是vgg
    # 这里的这个ls代表了large-scale，表示whether use large imag scale
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    # 如果这个文件存在，就读入文件的内容（配置），将它合并到/lib/model/utils/config.py文件中(也就是导入到变量cfg中)，这个文件是一个配置文件，里面有很多参数
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    # 将上面添加的配置，加入到config.py文件中，也就是导入到变量cfg中
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    # pprint()模块打印出来的数据结构更加完整，每行为一个数据结构，更加方便阅读打印输出结果。
    # 特别是对于特别长的数据打印，print()输出结果都在一行，不方便查看，而pprint()采用分行打印输出，所以对于数据结构比较复杂、
    # 数据长度较长的数据，适合采用pprint()打印方式。
    # 这个cfg变量是从cofig.py文件中弄来的，在上面的from import中import过来的
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # 进行数据的准备，combined_roidb(args.imdb_name)是数据准备的核心部分。
    # 返回的imdb是类pascal_voc的一个实例，后面只用到了其的一些路径，作用不大。roidb则包含了训练网络所需要的所有信息。下面看一下它的产生过程
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    print('{:d} roidb entries'.format(len(roidb)))

    # save-dir的默认值位models，net是特征提取器的类型，dataset是训练数据的名字，需要输入的
    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 得到采样器，用来定义采样的策略
    sampler_batch = sampler(train_size, args.batch_size)
    # dataset是数据集，得到的数据
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=True)
    # torch.utils.data.DataLoader是数据加载器，结合了数据集以及数据采样器
    # 在训练模型时使用到此函数，用来把训练数据分成多个小组，此函数每次抛出一组数据。直至把所有的数据都抛出。就是做一个数据的初始化
    # 输入参数
    # sampler参数定义了从数据集中提取数据的策略，因为要分成很多个batch，所以要提取数据，这个策略其实就是采样策略
    # num—workers代表了使用几个线程来加载数据
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    # torch.Tensor就是类似于创建变量，可以从numpy，或者python list创建
    # initilize the tensor holder here.
    # tensor([-2.2460e+11])这个应该是生成的一个随机数
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # 这里得到的这个fasterRcnn应该是得到的整个网络的变量
    # initilize the network here.
    if args.net == 'vgg16':
        # 这个imdb.classes 是一个元组,带上背景,总共有21类
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        # pdb是python用于调试代码的常用库
        # 如果在代码中看到
        # import pdb
        # pdb.set_trace()
        # 程序运行到这里就会暂停。
        pdb.set_trace()

    # 调用Vgg16类中的方法
    fasterRCNN.create_architecture()

    # 从配置文件中读取学习速率，并且设置args来覆盖，注意args设置的默认lr和cfg.TRAIN.LEARNING_RATE设置的相同
    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        # 只是将需要更新参数的记录在内
        if value.requires_grad:
            # 输出的这个key是一个字符串，而value是一个parameter
            if 'bias' in key:
                # 这样子加之后结果是[{},{},{},{}]
                # bias-decay是一个True或者False，而WEAIGHT-decay是一个数值
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.cuda:
        # 将所有的模型参数(parameters)和buffers赋值GPU
        fasterRCNN.cuda()

    # 将params放入到优化器里面，另外进行Adam优化的时候不需要进行momentum
    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # 这个应该是从断点处重新开始
    if args.resume:
        load_name = os.path.join(output_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    # 是不是使用多个GPU
    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    # 每个epoch的迭代次数，也就是有多少个batch
    iters_per_epoch = int(train_size / args.batch_size)

    # TensorBoard是一个可视化工具，它可以用来展示网络图、张量的指标变化、张量的分布情况等。
    # 这个用来判断是否使用tensorboard
    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # 设置为train模式，仅仅当模型中有Dropout和BatchNorm是才会有影响。
        # 我们 通常通过 module.train() 和 module.eval() 来切换模型的 训练测试阶段
        # setting to train mode
        fasterRCNN.train()
        # 用来累计loss,每一次要展示训练结果(n batch展示一次)的时候,就派上用场了.展示之后清零
        loss_temp = 0
        start = time.time()

        # 多少次epoch来做一次decay
        # 感觉这里不太爽,还需要手动进行decay,
        # 可以使用scheduler = lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)来进行自动调整lr
        if epoch % (args.lr_decay_step + 1) == 0:
            # lr-decay-gamma是learning rate decay ratio
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            # 直接乘以decay ratio
            lr *= args.lr_decay_gamma

        # 创建迭代器对象
        data_iter = iter(dataloader)
        # 迭代每个batch
        for step in range(iters_per_epoch):
            # 获取一个batch的数据
            data = next(data_iter)
            # with torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

            # zero_grad() 将module中的所有模型参数的梯度设置为0.
            fasterRCNN.zero_grad()
            # 调用这个方法之后,就会自动进入到forward函数里面执行前推
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            # 优化器的参数梯度设置为0
            optimizer.zero_grad()
            # 调用这个函数可以自动计算出来grad
            loss.backward()
            # 调用这个函数是用来解决梯度爆炸问题的,使用的方法叫做gradient clipping,要计算出一个norm参数乘进梯度里面
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            # 只有用了optimizer.step()，模型才会更新
            optimizer.step()

            # 这个参数的意思是number of iterations to display,这里的这个迭代指的是batch的次数,多少batch进行一次
            if step % args.disp_interval == 0:
                # 计算的是每次展示的时间间隔
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()

    # todo li
    #　我记得faster rcnn的论文里面说了训练的步骤,先训练a然后是b反正就很麻烦,为什么看这个代码感觉没有那么复杂