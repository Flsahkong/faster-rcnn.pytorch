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
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    # forward函数的输入为:(rpn_cls_prob.data, rpn_bbox_pred.data,im_info, cfg_key)
    #     分别是:softmax的输出2*9,bbox的输出4*9,以及im_info,和cfg-key当前状态train还是test
    # 这个模块的作用就是,综合RPN的那两条线,合并到一处,形成rois也就是roi
    # todo li 在rpn这里需要将预测的cls和bbox结合起来,成为一个输出,那怎么训练最后的分类器的时候没有将他们结合起来呢?难道这个工作是在测试的时候做的吗

    # 坐标说明,anchor的中心点坐标是(7.5,7.5),而且整个代码是以图片的中心点为坐标的.anchor放到图片上之后,其大小可能会超出图片的边界
    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        # FEAT_STRIDE,它的值位16,因为VGG16将原图缩小了16倍,这个是用来放大原图的
        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
            ratios=np.array(ratios))).float()
        # 这个值是9
        self._num_anchors = self._anchors.size(0)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold(阈值)
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        # input的内容是(rpn_cls_prob.data, rpn_bbox_pred.data,
        #                         im_info, cfg_key)

        # the first set of _num_anchors channels are bg probs,这个应该没有为什么,这是他们规定的,
        # 训练的时候,他们想认为第一个是fg,经过训练后,第一个就可以是fg
        # the second set are the fg probs,这里的得到的scores是fg大小是[1,9,37,忘了]
        scores = input[0][:, self._num_anchors:, :, :]
        bbox_deltas = input[1]
        im_info = input[2]
        cfg_key = input[3]

        # Number of top scoring boxes to keep before apply NMS to RPN proposals
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        # Number of top scoring boxes to keep after applying NMS to RPN proposals
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        # NMS threshold used on RPN proposals
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale),在这里值为8
        min_size = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)
        # 这里指的是feature的宽和高
        feat_height, feat_width = scores.size(2), scores.size(3)

        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        # numpy.meshgrid()——生成网格点坐标矩阵
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()


        # 这个A值是9
        A = self._num_anchors
        # 这个K的值是1850,shifts的大小为[1850,4]
        K = shifts.size(0)

        # 原来self._anchors的size是[9,4]就是9个anchor,之后大小还是[9,4],就是有了cuda参数
        self._anchors = self._anchors.type_as(scores)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        # 这时候得到的anchors就是[1,16650=1850*9=37*50*9=600/16 * 800/16 * 9,4] 给原始图像分了块,每一块都有9个anchors
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        # 经过这个步骤之后,原本是[1,50,37,36]变成了[1,50*37*9=16650,4]
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        # 经过这个步骤之后,变为[1,16650],原本是[1,50,37,9]
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via bbox transformations
        # 预测出来的proposal的四个坐标点分别是左上和右下
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        # 将我们得到的anchor进行修剪,不让他们超出图片
        proposals = clip_boxes(proposals, im_info, batch_size)
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it euqal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)

        # _, order = torch.sort(scores_keep, 1, True)

        # scores这个的size是[1,16650]
        scores_keep = scores
        # 这个的大小是[1,16650,4]
        proposals_keep = proposals
        # order的大小是[1,16650],这个order记录的是最高到最低的元素的index位置
        _, order = torch.sort(scores_keep, 1, True)
        # output的大小是[1,2000,5],值都为0,保留前两千个框
        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                # 提取最高的一些
                order_single = order_single[:pre_nms_topN]

            # 这两个的大小是[12000,1]和[12000,4]
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            # keep_idx_i保留的是需要留下的index
            keep_idx_i = nms(proposals_single, scores_single.squeeze(1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single
            # 最终output的大小是[1,2000,5],如果框没有2000个,那么后面的就全是0,保持为0
        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep
