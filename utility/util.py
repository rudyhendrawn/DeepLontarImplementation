"""
Source: https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/blob/8264dfba39a866998b8936a24133f41f12bfbdb7/util.py
"""
from __future__ import division

import cv2
import torch
import numpy as np

from torch.autograd import Variable

def unique(tensor: torch.tensor):
	tensor_np = tensor.cpu().numpy()
	unique_np = np.unique(tensor_np)
	unique_tensor = torch.from_numpy(unique_np)

	tensor_res = tensor.new(unique_tensor.shape)
	tensor_res.copy_(unique_tensor)

	return tensor_res

def predict_transform(prediction, inp_dim, anchors, num_classes, device=None):
	batch_size = prediction.size(0)
	stride = inp_dim // prediction.size(2)
	grid_size = inp_dim // stride
	bbox_attrs = 5 + num_classes
	num_anchors = len(anchors)

	prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
	prediction = prediction.transpose(1, 2).contiguous()
	prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

	anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

	# Sigmoid the centre_X, centre_Y. and object confidencce
	prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
	prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
	prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

	# Add the center offsets
	grid = np.arange(grid_size)
	a, b = np.meshgrid(grid, grid)

	x_offset = torch.FloatTensor(a).view(-1, 1)
	y_offset = torch.FloatTensor(b).view(-1, 1)

	if device is 'mps':
		x_offset = x_offset.to(device)
		y_offset = y_offset.to(device)
	else:
		x_offset = x_offset.cuda()
		y_offset = y_offset.cuda()

	x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

	prediction[:, :, :2] += x_y_offset

	# log space transform height and the width
	anchors = torch.FloatTensor(anchors)

	if CUDA:
		anchors = anchors.cuda()

	anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
	prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

	# Apply sigmoid activation to the class scores
	prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

	# Resize the detections map to the size of the input image
	prediction[:, :, :4] *= stride

	return prediction