import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import patches
from PIL import Image

class DeepLontarDataset(torch.utils.data.Dataset):
	"""
	Data pipeline class that reads images and its labels/annotations in YOLO format.
	All files are name in the following format:
	- JPEG images: <filename> .jpg, for instance: 1a.jpg, and
	- TXT annotations: <filename> .txt, for instance: 1a.txt
	Annotation files format using YOLO format, as follows:
	- <ID> <x> <y> <width> <height> for intance: 54 0.0068000 0.0833333 0.0160000 0.0733333, 
	where :
		- <ID> is the object class ID
		- <x> is x coordinate
		- <y> is y coordinate
		- <width> is width of the bounding box
		- <height> is height of the bounding box
	""" 
	def __init__(self, image_dir, label_dir):
		self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')])
		self.label_paths = sorted([os.path.join(label_dir, fname) for fname in os.listdir(label_dir) if fname.endswith('.txt')])

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		"""
		Args:
			idx (int): Index
		Returns:
			image (PIL Image): Image
			boxes (list): List of bounding boxes in the format [xmin, ymin, xmax, ymax, class_id]
		"""
		image_path = self.image_paths[idx]
		label_path = self.label_paths[idx]

		with open(label_path, 'r') as f:
			lines = f.readlines()
		
		boxes = []
		for line in lines:
			parts = line.strip().split()
			class_id = int(parts[0])
			x, y, width, height = map(float, parts[1:])
			xmin = (x - width / 2)
			ymin = (y - height / 2)
			xmax = (x + width / 2)
			ymax = (y + height / 2)
			boxes.append([xmin, ymin, xmax, ymax, class_id])

		# print(boxes)
		image = Image.open(image_path).convert('RGB')
		return image, boxes

	# Create a method to split the dataset into train and test set
	def split_dataset(self, split_ratio=0.8):
		"""
		Function to split the dataset into train and test set.
		Args:
			split_ratio (float): Ratio of the train set, for instance, 0.8 means 80% of the dataset is used for training
		Returns:
			train_dataset (DeepLontarDataset): Train dataset
			test_dataset (DeepLontarDataset): Test dataset
		"""
		train_size = int(len(self) * split_ratio)
		test_size = len(self) - train_size
		train_dataset, test_dataset = torch.utils.data.random_split(self, [train_size, test_size])
		return train_dataset, test_dataset
		
def visualize_dataset(dataset, idx):
	"""
	Function that can be used to visualize the dataset with its annotations.
	Args:
		dataset (DeepLontarDataset): Dataset to visualize
		idx (int): Index of the image to visualize

	Usage:
		visualize_dataset(dataset, 100)
	"""
	image, boxes = dataset[idx]
	image = np.array(image)

	fig, ax = plt.subplots(1)
	ax.imshow(image)
	for box in boxes:
		xmin, ymin, xmax, ymax, class_id = box
		# Scaling the bounding box to the image size so that it can be visible on the image
		xmin = xmin * image.shape[1]
		ymin = ymin * image.shape[0]
		xmax = xmax * image.shape[1]
		ymax = ymax * image.shape[0]
		width = xmax - xmin
		height = ymax - ymin
		bbox = plt.Rectangle((xmin, ymin), width, height, linewidth=0.5, edgecolor='r', facecolor='none')
		ax.add_patch(bbox)
		# ax.text(xmin, ymin, f'{class_id}', fontsize=10, color='r')
	plt.show()