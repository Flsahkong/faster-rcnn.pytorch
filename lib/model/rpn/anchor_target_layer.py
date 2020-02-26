from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()
        # Feature stride for RPN FEAT_STRIDE,它的值位16,因为VGG16将原图缩小了16倍,这个是用来放大原图的
        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        # 看这意思像是,允许框超过图片的边界多少
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors


        rpn_cls_score = input[0]
        # gt-boxes的大小是[1,20,5],通过查看这个变量得知,一个图片中最多含有的框为20个,
        # 如果小于20个,则其余的为0,即五个数字全部都是0
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        # 这个A值是9,K是1850,shifts的大小为[1850,4]
        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        # all_anchors的大小是[16650,4] 16650=37*50*9
        all_anchors = all_anchors.view(K * A, 4)

        total_anchors = int(K * A)
        # 这里也是限制anchor的边界的.keep的大小是[16650],里面标记了0和1
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))
        # 返回keep中非0元素的索引
        inds_inside = torch.nonzero(keep).view(-1)

        # 截取,从此之后,anchor的大小是[5944,4],5944是一张图片中anchor的个数
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # 这里创建的这几个的大小是[1,5944]
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        '''源自:https://zhuanlan.zhihu.com/p/64723237
               anchors: (N, 4)， 在图片内的所有原始anchors(映射到 网络输入图像上的)
               gt_boxes: (b, 20, 5) 每张图本身最多20个box
               overlaps: (b, N, 50), 表示每个anchor和每个gt_box的重叠面积的交并比IOU， 但这里并不是严格的交并比， 而是A^B/(AuB-A^B)
               如果不算batch的话， overlaps = 
               [[v11, v12, v13, ..., v120],
                [v21, v22, v23, ..., v220],
                ...
                [vN1, vN2, vN3, ..., vN20]]
               每一行表示一个anchor分别与50个gt box的IOU
        '''
        # 得到的大小是[1,5944,20],里面表示5944个anchor与每个gt框的IOU
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)

        # 找到每个anchor最大IOU的gt box的IOU
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        # 找到每个gt box的IOU最大的那些anchor
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        # 预设的值是Flase 意思是If an anchor statisfied by positive and negative conditions set to negative
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # IOU小于0.3的为negative , #RPN_NEGATIVE_OVERLAP=0.3的意思是 IOU < thresh: negative example
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        pdb.set_trace()
        # 如果gt box最大的anchor的IOU是0,就是说没有任何一个anchor与gt box相邻,改变他的值为1e-5(0.00001),这应该是为了下面
        # 进行对比的时候,不让0和0相同
        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        '''源自:https://zhuanlan.zhihu.com/p/64723237   
              overlaps: shape=(batch, N, 50)
              如果不算batch的话， overlaps = 
              [[v11, v12, v13, ..., v150],
               [v21, v22, v23, ..., v250],
               ...
               [vN1, vN2, vN3, ..., vN50]]
               每一行表示一个anchor分别与50个gt box的IOU
               gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps) = 
              [[vmax1, vmax2, vmax3, ..., vmax50],
               [vmax1, vmax2, vmax3, ..., vmax50],
               ...
               [vmax1, vmax2, vmax3, ..., vmax50]]    
               其中vmax1是v11到vN1中的最大一个， 其他同理。
               总共有N行。
              A.ep(B): A和B相同的元素的位置置1， 不相同的置0   
              overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps))：
              表示overlaps中gt box和哪个anchor的IOU最大， 那么其值就置为1， 其他的都置为0。
              那么这时候就会出现某些行全是0的情况， 也就是50个gt box的最大IOU对应的anchor最多只能是50个， 
              那么其他anchor所在的行的值都为0
              torch.sum(..., 2): 表示按行求和， 非全0的行sum的值就会大于0， 表示这个anchor是与gt boxes具有最大IOU的anchor中的一个
        '''
        # gt_max_overlaps原本是[batch,20]
        # gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps) 变为[batch,5944,20],5944是anchor的总数
        # A.ep(B): A和B相同的元素的位置置1， 不相同的置0
        # 经过这个ep之后,overlaps里面,和在dim=1的维度,和gt boxIOU最大的位置,会被置为1,其余为0
        # 等于我有5944行,20列. 经过这个步骤之后,每一列(5944个元素)里面有一个或者多个为1,其余为0
        # 最后的sum2表示按dim=2也就是20求和,其实就是按行求和.这时候有很多行会是0,一小部分行,会是1
        # 最后这个sum的作用是,找出和gt box的IOU最大的那些anchor
        # keep的size是[batch,5944]
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

        # 將与20个gt boxes具有最大IOU的anchor設置为正样本
        if torch.sum(keep) > 0:
            labels[keep>0] = 1

        # fg label: above threshold IOU
        # RPN_POSITIVE_OVERLAP这个值为0.7
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        # labels总结:
        # 他们从两个角度设置labels
        # 1. 从anchor的角度,如果anchor与20个gt box的IOU的最大值>=0.7,那么就设置这个anchor为1
        # 2. 从gt box角度,设置与每个gt box的IOU最大的那些anchor为1
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # RPN_FG_FRACTION是 Max number of foreground examples,是0.5
        # cfg.TRAIN.RPN_BATCHSIZE是256,Total number of examples
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        # labels有0,1,-1
        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                # 随机选择一部分前景(sum_fg-num_fg个)，置为-1(-1为无效框， 不是背景框)，只保留num_fg个前景
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

            # 得到背景的个数
            # num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                # 随机选择一部分背景景(sum_bg-num_bg个)，置为-1(-1为无效框， 不是背景框)，只保留num_bg个背景
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1
        # gt_boxes的大小是[1,20,5]
        # 我看代码的时候,batch是1,得到的offset是[0],如果batch是3,得到的offset就是[0,20,40]
        offset = torch.arange(0, batch_size)*gt_boxes.size(1)

        # argmax_overlaps， shape=(batch, N=5944), 是每个anchor最大IOU的gt box的index,index的值是0-19
        # argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps):
        # offset.view(batch_size, 1),将横着的[0,20,40,...]改为竖着的,大小是[batch,1]
        # 将每个anchor最大IOU的gt box的index分别加上0,20,40,...,(batch-1)*20
        # 结果argmax_overlaps shape还是(batch, N)
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)

        # gt_boxes.view(-1,5) shape = (batch*20, 5)
        # argmax_overlaps.view(-1) shape=(batch*N)
        # 所以gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :] 则表示选择出与每个anchor最大IOU的gt box的五个数值
        # gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5)  shape=(batch, N, 5)
        # anchors: (N, 4)， 在图片内的所有原始anchors(映射到 网络输入图像上的)
        # bbox_targets： (b, N, 4), 每个anchor与其最大IOU的gt box的平移and缩放比例
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        # https://www.zhihu.com/question/65587875
        # 用来设置正样本回归 loss 的权重，默认为 1（负样本为0，即可以区分正负样本是否计算 loss）。
        # 原文中，正样本计算回归 loss，负样本不计算（可认为 loss 为 0）
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        # RPN_POSITIVE_WEIGHT=-1.0 Set to -1.0 to use uniform example weighting
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # 前景和背景样本总数(应该等同于正负样本总数)
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        # bbox_outside_weights用来平衡 RPN 分类 Loss 和回归 Loss 的权重。
        '''
        查看图片 ./myNote/bbox outside weights.png
        '''
        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        # inds_inside： (batch, N), 所有在图像范围内的anchor的index
        # total_anchors： weight*height*9, 所有的anchor数
        # labels: (batch, N), 所有在图像范围内的anchor的label
        # return labels: shape=(batch, weight*height*9), 所有的anchor的label, 不在图像范围的置为-1
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        # labels的大小是[batch,1,333=37*9,50],里面记录了0,1,-1,就是前景的anchor,背景的anchor和无关的anchor
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        # bbox_targets的大小是[1, 36, 37, 50],表示每个anchor与其最大IOU的gt box的平移and缩放比例
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        # bbox_inside_weights的大小是[1, 36, 37, 50],表示# todo li
        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)
        # bbox_outside_weights的大小是[1, 36, 37, 50],表示# todo li
        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
