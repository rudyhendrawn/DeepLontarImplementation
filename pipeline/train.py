import os
import torch
import editdistance

from model_architecture.darknet import YOLO


class Trainer:
	def __init__(self, train_loader, test_loader, model, optimizer, loss_fn, device):
		# self.dataset = dataset
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.model = model
		self.optimizer = optimizer
		self.loss_fn = loss_fn
		self.device = device

	def train(self, num_epochs, batch_size, learning_rate):
		# Define data loader
		# dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

		# Train the model
		for epoch in range(num_epochs):
			for batch_idx, (images, labels) in enumerate(self.train_loader):
				# Move images and labels to device
				images = images.to(self.device)
				labels = labels.to(self.device)
				
				# Zero out the gradients
				self.optimizer.zero_grad()

				# Forward pass
				predictions = self.model(images)
				loss = self.loss_fn(predictions, labels)

				# Backward pass
				loss.backward()
				self.optimizer.step()

				# Print training progress
				if batch_idx % 10 == 0:
					print(f'Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(self.train_loader)} Loss: {loss.item():.4f}')

	def evaluate(self, batch_size):
		# Define data loader
		# dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

		# Initialize counters for evaluation metrics
		total_chars = 0
		total_correct_chars = 0
		total_words = 0
		total_correct_words = 0
		total_char_errors = 0
		total_word_errors = 0

		# Evaluate the model
		self.model.eval()
		with torch.no_grad():
			for images, labels in self.test_loader:
				# Move images and targets to device
				images = images.to(self.device)
				labels = labels.to(self.device)

				# Forward pass
				self.test_loader = self.model(images)
				predicted_boxes = self.model.get_output_boxes(	predictions, 
																self.test_loader.input_shape,
																self.test_loader.anchors,
																self.test_loader.num_classes
								)

				# Convert predicted boxes and ground truth labels to text
				predicted_text = self.test_loader.decode_output_boxes(predicted_boxes)
				label_text = self.test_loader.decode_target_boxes(labels)

				# Compute evaluation metrics
				for predicted, label in zip(predicted_text, label_text):
					# Character accuracy
					total_chars += len(label)
					total_correct_chars += sum([p == t for p, t in zip(predicted, label)])

					# Word accuracy
					predicted_words = predicted.split()
					label_words - target.split()
					total_words += len(target_words)
					total_correct_words += sum([p == t for p, t in zip(predicted_words, label_words)])

					# Character error rate
					total_char_errors += editdistance.eval(predicted, label)

					# Word error rate
					total_word_errors += editdistance.eval(predicted_words, label_words)
			
			# Compute evaluation metrics as percentages
			char_accuracy = total_correct_chars / total_chars * 100
			word_accuracy = total_correct_words / total_words * 100
			char_error_rate = total_char_errors / total_chars * 100
			word_error_rate = total_word_errors / total_words * 100
			# accuracy = total_correct / total_images
			# print(f'Accuracy: {accuracy:.4f}') 

			# Print evaluation metrics
			print(f'Character accuracy: {char_accuracy:.4f}')
			print(f'Word accuracy: {word_accuracy:.4f}')
			print(f'Character error rate: {char_error_rate:.4f}')
			print(f'Word error rate: {word_error_rate:.4f}')
