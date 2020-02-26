from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        # Feature stride for RPN FEAT_STRIDE,它的值位16,因为VGG16将原图缩小了16倍,这个是用来放大原图的
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # 定义一个3*3的卷积核,步长为1,对应RPN的第一个卷积层
        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer,计算出来的结果为18,对应往上走的步骤
        # 这里的18的意思是,feature map的每个点都有9个anchor,每个anchor又都有两个结果,positive和negative
        # 这个positive就是前景fg,而negative就是背景bg
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer,计算出来的结果为36,对应下面那一条路
        # 这里的bbox预测的是tx,ty,tw,th并不是真正的边框的坐标,需要将这几个t与anchor进行运算,得到真正的边框值
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    # 静态方法不用使用self作为参数
    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)


        # inplace的操作是,计算出来的结果覆盖,
        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # 先经过上面的3*3,然后经过这个1*1
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        # 在经过reshape层
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        # 使用softmax进行二分类,看看feature map上面每个点的9个anchor里面,哪个是包含了物体(fg前景)的
        # 第二个参数叫做dim,是指进行softmax的维度,这个1表示第二维
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        # li = rpn_cls_prob_reshape.view(rpn_cls_prob_reshape.size()[0],rpn_cls_prob_reshape.size()[2]
        # ,rpn_cls_prob_reshape.size()[3],
        #                                 rpn_cls_prob_reshape.size()[1])
        #
        # print(li)
        # print(li.size())
        # print(rpn_cls_prob_reshape.size())
        # get rpn offsets to the anchor boxes
        # 经过上面的注释代码表明,softmax输出的每个anchor的值为0-1之间,并且他们的和并不是1,为什么不是1?
        # 经过测试发现,不能简单地,使用view来进行.view 和reshape得到的结果是一样的,想交换维度,需要用到torch.transpose

        # 经过1*1的层,然后结果为4*9,todo li 这个4具体指的是什么
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        # rois的大小为[1,2000,5]坐标在三维的最后四个,并且坐标点是左上和右下角
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        import pdb
        pdb.set_trace()

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            # 这个rpn-data是在计算loss的时候用上了
            # 在这里使用的是rpn_cls_score,这个是第一条线1*1卷积的输出,输出为18,没有进行softmax
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box
