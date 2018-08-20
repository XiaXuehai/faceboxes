# coding:utf-8
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
	num_classes = 2
	def __init__(self):
		super(MultiBoxLoss,self).__init__()

	def cross_entropy_loss(self, x, y):
		x = x.detach()
		y = y.detach()
		xmax = x.max()
		log_sum_exp = torch.log(torch.sum(torch.exp(x-xmax), 1, keepdim=True)) + xmax
		return log_sum_exp - x.gather(1, y)

	def hard_negative_mining(self,conf_loss,pos):
		'''
		conf_loss [N*21482,]
		pos [N,21482]
		return negative indice
		'''
		batch_size, num_boxes = pos.size()
		conf_loss[pos.view(-1,1)] = 0  #去掉正样本,the rest are neg conf_loss
		conf_loss = conf_loss.view(batch_size,-1)

		_,idx = conf_loss.sort(1,descending=True)
		_,rank = idx.sort(1) # the rank means origin number's rank in the position

		num_pos = pos.long().sum(1,keepdim=True)
		num_neg = torch.clamp(3*num_pos, max=num_boxes-1)

		neg = rank < num_neg.expand_as(rank) # torch.ByteTensor
		return neg

	def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
		'''
		loc_preds    [batch,21824,4]
		loc_targets  [batch,21824,4]
		conf_preds   [batch,21824,2]
		conf_targets [batch,21824]
		'''
		pos = conf_targets > 0  # the place > 0 is a face, to torch.uint8
		num_pos = pos.sum(1, keepdim=True) # num of faces in one image over the batch_size
		num_matched_boxes = num_pos.sum().item() # num of all faces
		if num_matched_boxes == 0:
			return torch.Tensor([0], requires_grad=True)

		# got the loc loss
		pos_mask1 = pos.unsqueeze(2).expand_as(loc_preds)
		pos_loc_preds = loc_preds[pos_mask1].view(-1,4) # filter the pos face
		pos_loc_targets = loc_targets[pos_mask1].view(-1,4)
		loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets) * 4. # 4 is get the sum of four loc

		# got the conf loss
		# is not cross_entropy_loss,is log_sum_exp,just to get neg
		conf_loss = self.cross_entropy_loss(conf_preds.view(-1,self.num_classes),
									conf_targets.view(-1,1))
		neg = self.hard_negative_mining(conf_loss, pos) # (16*21824, (16,21824))
		pos_mask = pos.unsqueeze(2).expand_as(conf_preds)
		neg_mask = neg.unsqueeze(2).expand_as(conf_preds)

		mask = pos_mask+neg_mask
		pos_and_neg = pos+neg

		preds = conf_preds[mask].view(-1,self.num_classes)
		targets = conf_targets[pos_and_neg]
		conf_loss = F.cross_entropy(preds, targets, reduction='sum') / num_matched_boxes

		#print('loc_loss:%f conf_loss:%f, pos_num:%d' % (loc_loss.item(), conf_loss.item(), num_matched_boxes))
		return loc_loss + conf_loss
