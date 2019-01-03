# coding:utf-8
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def init_model(m):
	if isinstance(m, nn.Conv2d):
		nn.init.xavier_uniform_(m.weight)
	elif isinstance(m, nn.BatchNorm2d):
		nn.init.constant_(m.weight, 1)
		nn.init.constant_(m.bias, 0)
	return m

class MultiBoxLayer(nn.Module):
	num_classes = 2
	num_anchors = [21,1,1]
	in_planes = [128,256,256]

	def __init__(self):
		super(MultiBoxLayer,self).__init__()
		
		self.loc_layers = nn.ModuleList()
		self.conf_layers = nn.ModuleList()
		for i in range(len(self.in_planes)):
			m0 = nn.Conv2d(self.in_planes[i], self.num_anchors[i] * 4, kernel_size=3, padding=1)
			m0 = init_model(m0)
			m1 = nn.Conv2d(self.in_planes[i], self.num_anchors[i] * 2, kernel_size=3, padding=1)
			m1 = init_model(m1)
			self.loc_layers.append(m0)
			self.conf_layers.append(m1)

	def forward(self,xs):
		'''
		xs:list of 之前的featuremap list
		retrun: loc_preds: [N,21824,4]
				conf_preds:[N,24824,2]
		'''
		y_locs=[]
		y_confs = []
		for i,x in enumerate(xs):
			y_loc = self.loc_layers[i](x) # N,anhors*4,H,W
			N = y_loc.size(0)
			y_loc = y_loc.permute(0,2,3,1).contiguous()
			y_loc = y_loc.view(N,-1,4)
			y_locs.append(y_loc)

			y_conf = self.conf_layers[i](x)
			y_conf = y_conf.permute(0,2,3,1).contiguous()
			y_conf = y_conf.view(N,-1,2)
			y_confs.append(y_conf)
			
		loc_preds = torch.cat(y_locs,1)
		conf_preds = torch.cat(y_confs,1)

		return loc_preds, conf_preds