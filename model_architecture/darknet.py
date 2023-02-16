"""
Create a class that Implements YOLOv3 model architecture
The input model is the image data and its annotations (bounding boxes) and class_id in format [xmin, ymin, xmax, ymax, class_id]
Source: https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/blob/master/darknet.py
"""
import torch
import torchvision

class YOLO(torch.nn.Module):
	def __init__(self, num_classes, anchors=None):
		super().__init__()
		self.darknet = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).model.backbone

		self.detect = torch.nn.Sequential(
			torch.nn.Conv2d(1024, 512, kernel_size=1),
			torch.nn.BatchNorm2d(512),
			torch.ReLU(inplace=True),
			torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			torch.BatchNorm2d(1024),
			nn.ReLU(inplace=True),
			torch.nn.Conv2d(1024, 512, kernel_size=1),
			torch.nn.BatchNorm2d(512),
			torch.ReLU(inplace=True),
			torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			torch.nn.BatchNorm2d(1024),
			torch.ReLU(inplace=True),
			torch.nn.Conv2d(1024, num_classes*(5+len(anchors)), kernel_size=1)
		)
		self.anchors = anchors
		self.num_classes = num_classes

	def forward(self, x):
		x = self.darknet(x)
		x = self.detect(x)
		return x

	def get_output_boxes(self, output_tensor, input_shape, anchors, num_classes):
		"""
		Args:
			output_tensor (torch.Tensor): Output tensor of the model
			input_shape (tuple): Shape of the input image
			anchors (list): List of anchors
			num_classes (int): Number of classes
		Returns:
			boxes (list): List of bounding boxes in the format [xmin, ymin, xmax, ymax, class_id, confidence]
		"""
		# Compute the grid size for each detection scale
		grid_sizes = [(input_shape[0] // s, input_shape[1] // s) for s in [32, 16, 8]]

		# Reshape the output tensor to have [batch_size, num_anchors, num_classes+5, grid_size, grid_size]
		batch_size, _, grid_size_y, grid_size_x = output_tensor.size()
		output_tensor = output_tensor.view(batch_size, len(anchors), num_classes+5, grid_size_y, grid_size_x)

		# Apply sigmoid to the center coordinates and objectness score
		output_tensor[..., :2] = torch.sigmoi(output_tensor[..., :2])
		output_tensor[..., 4] = torch.sigmoid(output_tensor[..., 4])

		# Compute the bounding box coordinates
		bbox_xy = output_tensor[..., :2]
		bbox_wh = output_tensor[..., 2:4]
		bbox_xy += torch.arange(grid_size_x, device=output_tensor.device, dtype=torch.float).view(1, 1, -1, 1)
		bbox_xy += torch.arange(grid_size_y, device=output_tensor.device, dtype=torch.float).view(1, 1, 1, -1)
		bbox_xy *= 32
		anchors = torch.tensor(anchors, device=output_tensor.device)
		bbox_wh *= anchors.view(1, len(anchors), 1, 1)
		bbox_x1y1 = bbox_xy - bbox_wh / 2
		bbox_x2y2 = bbox_xy + bbox_wh / 2
		output_tensor[..., :4] = torch.cat([bbox_x1y1, bbox_x2y2], dim=-1)

		# Convert the bounding box coordinates to the annotation format
		output_tensor[..., 0] += 1	# Add 1 to the class ID to match the annotation format
		output_tensor[..., :2] *= input_shape[::-1]	# Convert center coordinates to image coordinates
		output_tensor[..., 2:4] *= input_shape[::-1]	# Convert width and height to image coordinates

		# Reshape the output tensor to [batch_size, num_anchors * grid_size * grid_size, num_classes + 5]
		output_tensor = output_tensor.permute(0, 3, 4, 1, 2).contiguous()
		output_tensor = output_tensor.view(batch_size, -1, num_classes+5)

		return output_tensor