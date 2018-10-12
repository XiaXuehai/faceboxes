#  coding:utf-8

import torch
import math
import itertools
import cv2
import numpy as np

class DataEncoder:
    def __init__(self):
        '''
        compute default boxes
        '''
        scale = 1024.
        sizes = [s / scale for s in (32, 256, 512)] # 当32改为64时，achor与label匹配的正样本数目更多
                                                    # anchor's size.In the paper, the first output layer size is (32, 64, 128)
        aspect_ratios = ((1,2,4), (1,), (1,)) # more understanding
        feature_map_sizes = (32, 16, 8)

        density = [[-3,-1,1,3],[-2,2],[0]] # density for output layer1
        
        num_layers = len(feature_map_sizes)
        boxes = []
        for i in range(num_layers):   ### 3 times
            fmsize = feature_map_sizes[i]
            for h,w in itertools.product(range(fmsize), repeat=2): ###   featuresize.pow(2)
                cx = (w + 0.5)/feature_map_sizes[i] # every grid value of centerX
                cy = (h + 0.5)/feature_map_sizes[i]

                s = sizes[i]
                for j,ar in enumerate(aspect_ratios[i]): ### 3 , 1, 1
                    if i == 0:
                        for dx,dy in itertools.product(density[j], repeat=2):   ### 21 = 4*4 + 2*2 + 1
                            boxes.append((cx+dx/8.*s*ar, cy+dy/8.*s*ar, s*ar, s*ar)) # add the offset
                    else:
                        boxes.append((cx, cy, s*ar, s*ar))
        
        self.default_boxes = torch.Tensor(boxes)

    def iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].

        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max( # left top
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min( # right bottom
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def nms(self, bboxes, scores, threshold=0.5):
        '''
        bboxes(tensor) [N,4]
        scores(tensor) [N,]
        '''
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]
        areas = (x2-x1) * (y2-y1)

        _,order = scores.sort(0,descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr<=threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]
        return torch.LongTensor(keep)

    def encode(self, boxes, classes, threshold=0.35):
        '''
        boxes:[num_obj, 4]
        default_box (x1,y1,w,h)
        return:boxes: (tensor) [num_obj,21824,4]
        classes:class label [obj,]
        '''
        boxes_org = boxes

        default_boxes = self.default_boxes  # [21824,4]

        iou = self.iou(
            boxes,
            torch.cat([default_boxes[:, :2] - default_boxes[:, 2:] / 2,  # (x0,y0,w,h) => (x1,y1,x2,y2)
                       default_boxes[:, :2] + default_boxes[:, 2:] / 2], 1))  # iou_size = (num_obj, 21824)

        # find max iou of each face in default_box
        max_iou, max_iou_index = iou.max(1)
        # find max iou of each default_box in faces
        iou, max_index = iou.max(0)

        # ensure every face have a default box,I think is no use.
        # max_index[max_iou_index] = torch.LongTensor(range(num_obj))

        boxes = boxes[max_index]  # [21824,4] 是图像label, use conf to control is or not.
        variances = [0.1, 0.2]
        cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - default_boxes[:, :2]  # [21824,2]
        cxcy /= variances[0] * default_boxes[:, 2:]
        wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]  # [21824,2]  为什么会出现0宽度？？
        wh = torch.log(wh) / variances[1]  # Variable

        inf_flag = wh.abs() > 10000
        if (inf_flag.sum().item() is not 0):
            print('inf_flag has true', wh, boxes)
            print('org_boxes', boxes_org)
            print('max_iou', max_iou, 'max_iou_index', max_iou_index)
            raise 'inf error'

        loc = torch.cat([cxcy, wh], 1)  # [21824,4]
        conf = classes[max_index]  # 其实都是1 [21824,]
        conf[iou < threshold] = 0  # iou小的设为背景
        #conf[max_iou_index] = 1    # 这么设置有问题，loc loss 会导致有inf loss，从而干扰训练，
                                    # 去掉后，损失降的更稳定些，是因为widerFace数据集里有的label
                                    # 做的宽度为0，但是没有被滤掉，是因为max(1)必须为每一个object选择一个
                                    # 与之对应的default_box，需要修改数据集里的label。

        return loc, conf

    def decode(self, loc, conf):
        '''
        將预测出的 loc/conf转换成真实的人脸框
        loc [21842,4]
        conf [21824,2]
        '''
        variances = [0.1, 0.2]
        cxcy = loc[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        wh  = torch.exp(loc[:,2:] * variances[1]) * self.default_boxes[:,2:]
        boxes = torch.cat([cxcy-wh/2,cxcy+wh/2],1) #[21824,4]
        
        # conf[:,0] means no face
        # conf[:,1] means face
        # filter face by a value of 0.4
        #conf[:,0] = 0.4

        max_conf, labels = conf.max(1) #[21842,1]

        if labels.long().sum().item() is 0:
            # no face in image
            return torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

        ids = labels.nonzero().squeeze(1)
        # print('ids', ids)
        # print('boxes', boxes.size(), boxes[ids])

        keep = self.nms(boxes[ids],max_conf[ids])

        return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]

