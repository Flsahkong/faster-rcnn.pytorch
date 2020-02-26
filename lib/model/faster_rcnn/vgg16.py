# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb


class vgg16(_fasterRCNN):
    # 这里的这个pretrain参数其实没什么作用，在创建类的时候，传入的是True，已经下载了预训练好的模型
    # 这里的class_agnostic的意思是whether perform class_agnostic bbox regression，
    # 就是说在进行bbox回归的时候，是输出一个框，还是输出n-classes个框
    def __init__(self, classes, pretrained=False, class_agnostic=False):
        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        # 512是得到的feature map的维度
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        # 这里是加载已经写好的模型vgg16，并且不要预训练模型，因为预训练模型已经准备好了（在data文件夹）
        vgg = models.vgg16()
        # 加载vgg16的训练参数，这是预训练的模型，后面还要接着训练
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        # 下面给出vgg16的模型输出信息
        '''
        VGG(
          (features): Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace)
            (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): ReLU(inplace)
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (6): ReLU(inplace)
            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (8): ReLU(inplace)
            (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (11): ReLU(inplace)
            (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (13): ReLU(inplace)
            (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (15): ReLU(inplace)
            (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (18): ReLU(inplace)
            (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (20): ReLU(inplace)
            (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (22): ReLU(inplace)
            (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (25): ReLU(inplace)
            (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (27): ReLU(inplace)
            (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (29): ReLU(inplace)
            (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          )
          (classifier): Sequential(
            (0): Linear(in_features=25088, out_features=4096, bias=True)
            (1): ReLU(inplace)
            (2): Dropout(p=0.5)
            (3): Linear(in_features=4096, out_features=4096, bias=True)
            (4): ReLU(inplace)
            (5): Dropout(p=0.5)
            (6): Linear(in_features=4096, out_features=1000, bias=True)
          )
        )
        '''
        # 这个classifier是指上面的最后的三个linear层
        # vgg.classifier._modules的输出
        # OrderedDict([('0', Linear(in_features=25088, out_features=4096, bias=True)),
        # ('1', ReLU(inplace)), ('2', Dropout(p=0.5)), ('3', Linear(in_features=4096, out_features=4096, bias=True)),
        # ('4', ReLU(inplace)), ('5', Dropout(p=0.5)), ('6', Linear(in_features=4096, out_features=1000, bias=True))])
        # nn.Sequential(*list(vgg.classifier._modules.values())[:-1])的输出
        # Sequential(
        #   (0): Linear(in_features=25088, out_features=4096, bias=True)
        #   (1): ReLU(inplace)
        #   (2): Dropout(p=0.5)
        #   (3): Linear(in_features=4096, out_features=4096, bias=True)
        #   (4): ReLU(inplace)
        #   (5): Dropout(p=0.5)
        # )
        # 由此可见，这句话是把原来的最后一层的linear给删了
        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # 这一句是把vgg的最后一层的Maxpool层给删除了
        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # 给前十层设置不更新参数
        # Fix the layers before conv3:
        for layer in range(10): # i为0-9
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        # 弄了两个模块，一个叫做RCNN-base，是特征提取层。一个叫做RCNN-top，是最后的分类器
        self.RCNN_top = vgg.classifier

        # 分类器的最后一层输入输出都是4096，所以这一层相当于是最后的分类层，分为n-classes种类型
        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        # 判断进行最后的回归的时候，要不要输出n-classes的分类数据
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

        # todo li 这里有一个很重要的问题：
        # 看上面的cls-score的情况，应该是相当于一张图片只输出一个物体，那如果一个图片有很多个物体呢

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        # 在这里调用了vgg.classifier,先经过了两个线性层,输出特征
        fc7 = self.RCNN_top(pool5_flat)

        return fc7
