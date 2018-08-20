# coding:utf-8
'''
txt描述文件 image_name.jpg num x y w h 1 x y w h 1 这样就是说一张图片中有两个人脸
'''
import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2

from encoderl import DataEncoder
from augmentations import SSDAugmentation

class ListDataset(data.Dataset):

    def __init__(self, root, list_file, train, transform):
        print('data init')
        self.image_size = 1024
        self.root=root
        self.train = train
        self.transform=transform
        self.fnames = [] # list: image name
        self.boxes = []
        self.labels = []
        self.small_threshold = 20./self.image_size  # face that small than threshold will be ignored
                                                    # it's 20 in the paper
        self.data_encoder = DataEncoder()

        with open(list_file) as f:
            lines  = f.readlines()

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_faces = int(splited[1])
            box=[]
            label=[]
            for i in range(num_faces):
                x = float(splited[2+5*i])
                y = float(splited[3+5*i])
                w = float(splited[4+5*i])
                h = float(splited[5+5*i])
                c = int(splited[6+5*i])
                box.append([x,y,x+w,y+h])
                label.append(c)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.tensor(label))
        self.num_samples = len(self.boxes) # num of images

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname))

        assert img is not None

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.train:
            # img, boxes, labels = self.random_crop(img, boxes, labels)
            # img = self.random_bright(img)
            # img, boxes = self.random_flip(img, boxes)

            img, boxes, labels = SSDAugmentation()(img, boxes, labels)



        h,w,_ = img.shape
        img = cv2.resize(img,(self.image_size,self.image_size))

        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        for t in self.transform:
            img = t(img)

        loc_target,conf_target = self.data_encoder.encode(boxes,labels)

        return img,loc_target,conf_target

    def __len__(self):
        return self.num_samples

    def random_getim(self):
        idx = random.randrange(0, self.num_samples)
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname))
        box = self.boxes[idx].clone()
        label = self.labels[idx].clone()

        return img, box, label

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_crop(self, im, boxes, labels):
        imh, imw, _ = im.shape
        short_size = min(imw, imh)
        while True:
            # choose one crop pitch
            mode = random.randint(0, 4)
            for _ in range(10):
                if mode == 0:
                    w = short_size
                else:
                    w = random.randrange(int(0.3*short_size), short_size)
                h = w

                x = random.randint(0, imw - w)
                y = random.randint(0, imh - h)
                roi = torch.Tensor([[x, y, x+w, y+h]])

                center = (boxes[:,:2] + boxes[:,2:]) / 2
                roi2 = roi.expand(len(center), 4)
                mask = (center > roi2[:,:2]) & (center < roi2[:,2:])
                mask = mask[:,0] & mask[:,1]
                if not mask.any():
                    im, boxes, labels = self.random_getim()
                    imh, imw, _ = im.shape
                    short_size = min(imw, imh)
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
                img = im[y:y+h,x:x+w,:]
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)

                boxes_uniform = selected_boxes / torch.Tensor([w,h,w,h]).expand_as(selected_boxes)
                boxwh = boxes_uniform[:,2:] - boxes_uniform[:,:2]
                mask = (boxwh[:,0] > self.small_threshold) & (boxwh[:,1] > self.small_threshold)
                if not mask.any():
                    # crop image have none box bigger than small_threshold
                    im, boxes, labels = self.random_getim()
                    imh, imw, _ = im.shape
                    short_size = min(imw,imh)
                    continue
                selected_boxes_selected = selected_boxes.index_select(0, mask.nonzero().squeeze(1))
                selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))
                return img, selected_boxes_selected, selected_labels

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im




if __name__ == '__main__':
    file_root = '/home/xxh/Desktop/learning_pytorch/faceboxes-master/'
    list_file = 'data/val_rewrite.txt'
    train_dataset = ListDataset(root=file_root,
                                list_file=list_file,
                                train=True,
                                transform = [transforms.ToTensor()] )

    print('the dataset has %d image' % (len(train_dataset)))
    image, boxes,labels = train_dataset[111]


