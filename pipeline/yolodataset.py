import torch
import numpy as np

class YoloDataset(torch.Dataset):
	def __init__(self, image_dir, label_dir, input_shape, anchors, num_classes, transform=None):
		self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')])
		self.label_paths = sorted([os.path.join(label_dir, fname) for fname in os.listdir(label_dir) if fname.endswith('.txt')])
		self.input_shape = input_shape
		self.anchors = np.array(anchors).reshape((-1, 2))
		self.num_classes = num_classes
		self.transform = transform
		self.class_map = {i: i for i in range(num_classes)}
		self.grid_sizes = [(input_shape[0] // s, input_shape[1] // s) for s in [32, 16, 8]]
	
	def decode_output_boxes(self, boxes):
		# Convert YOLO-format bounding boxes to pixel coordinates
		boxes = self._yolo_to_pixel(boxes, self.input_shape)

		# Convert bounding boxes to text
		texts = []
		for i in range(boxes.shape[0]):
			texts = ''
			for j in range(boxes.shape[1]):
				box  = boxes[i, j, :]
				if box.sum() == 0:
					continue
				x1, y1, x2, y2 = box
				center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
				width, height = x2 - x1, y2 - y1
				text += str(self.class_map[j]) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(width) + ' ' + str(height) + '\n'
				text.append(text)
		return texts

	def decode_target_boxes(self, boxes):
		# Convert YOLO-format bounding boxes to pixel coordinates
		boxes = self._normalize_to_pixel(boxes, self.input_shape)

		# Convert bounding boxes to text
		texts = []
		for i in range(boxes.shape[0]):
			text = ''
			for j in range(boxes.shape[1]):
				box  = boxes[i, j, :]
				if box.sum() == 0:
					continue
				x1, y1, x2, y2 = box
				center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
				width, height = x2 - x1, y2 - y1
				text += str(self.class_map[j]) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(width) + ' ' + str(height) + '\n'
			texts.append(text)
		return texts

	def _yolo_to_pixel(self, boxes, input_shape):
		num_boxes, num_anchors, num_classes = boxes.shape[0], boxes.shape[1], boxes.shape[4]
		grid_size = input_shape[0] // boxes.shape[2]
		boxes[..., :2] = (boxes[..., :2] + self.grid) * grid_size
		boxes[..., 2:4] = np.exp(boxes[..., 2:4]) * self.anchors / input_shape[::-1]
		boxes[..., :4] = boxes[..., :4][..., ::-1]
		boxes[..., :4] *= self.input_shape + (self.input_shape == boxes.shape[2:]) 
		return boxes.astype(int)

	def _normalize_to_pixel(self, boxes, input_shape):
		boxes[..., :4] *= input_shape + (input_shape == boxes.shape[2:])
		return boxes.astype(int)