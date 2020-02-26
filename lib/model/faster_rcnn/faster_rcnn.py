import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

# nn.module是实现自己的网络必须要继承的类，需要实现这个类的forward方法。
class _fasterRCNN(nn.Module):
    """ faster RCNN """
    # 这里的class_agnostic的意思是whether perform class_agnostic bbox regression，
    # 就是说在进行bbox回归的时候，是输出一个框，还是输出n-classes个框
    def __init__(self, classes, class_agnostic):
        # 继承这个Module类，这个super函数一定要调用
        super(_fasterRCNN, self).__init__()
        # classes一共有21类,带上了背景
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        # 输入进去的cfg.POOLING_SIZE的大小是7
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        # im_data的大小是[1,3,600,800],这个1其实是图片总数,在这里就是指batch的大小
        batch_size = im_data.size(0)
        # todo li 这个num-box到底是个什么东西,im-info是什么东西
        # im_info 的大小是[600,800,1.6]
        # gt_boxes的大小是[1,20,5]
        # num_boxes的大小是[1],数值是3
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # rios应该是region of interest
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:

            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            # 所有的大小都是fg_rois_per_image=256,都是[batch,256,5或4]
            # rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            # 注意! 从这之后,所有的结果就开始了
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            # 下面这个语句在运行train的时候,没有执行这里的句子
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))


        # 这个_head_to_tail函数在vgg16.py文件里面,在这个步骤中,经过了两层线性层,输出的pooled_feat就可以直接进行分类和回归了
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # 这个RCNN_bbox_pred也在vgg16文件中,计算出预测的框
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # RCNN_cls_score也在vgg16中,经过一个线性层,输出20个类别的分数
        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        # 经过softmax输出每个类别的概率
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            # rois_label是每个rois与gt box形成的labels
            # 从这个loss可以看出来,预测的cls_score是针对rois进行的
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            # 和上面一样,bbox也是针对rois进行的,预测的是针对每个rois的tx,ty,tw,th
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # 进行几个卷积层的数据初始化,高斯分布,均值和方差
        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        # 真神奇啊，外部的vgg16对象的self传了进来，所以可以调用子类vgg16里面的_init_modules方法
        self._init_modules()
        self._init_weights()
