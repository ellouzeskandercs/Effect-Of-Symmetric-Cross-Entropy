""" Inspired by: Copyright (c) 2017 pat-coady: https://github.com/pat-coady/tiny_imagenet """

import glob
import re
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class config(object):
	batch_size = 64
	num_epochs = 50

def load_filenames_labels(mode):
	"""Gets filenames and labels Args
	mode:
		'train' or 'val'
		(Directory structure and file naming different for
		train and val datasets)
	Returns:
		list of tuples: (jpeg filename with path, label)
	"""
	label_dict, class_description = build_label_dicts()
	filenames_labels = []
	if mode == 'train':
		filenames = glob.glob('datasets/tiny-imagenet-200/train/*/images/*.JPEG')
		for filename in filenames:
			match = re.search(r'n\d+', filename)
			label = str(label_dict[match.group()])
			filenames_labels.append((filename, label))
	elif mode == 'val':
		with open('datasets/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
			for line in f.readlines():
				split_line = line.split('\t')
				filename = 'datasets/tiny-imagenet-200/val/images/' + split_line[0]
				label = str(label_dict[split_line[1]])
				filenames_labels.append((filename, label))

	print(type(filenames_labels))
	print(len(filenames_labels))

	return filenames_labels

def build_label_dicts():
	"""Build look-up dictionaries for class label, and class description
	Class labels are 0 to 199 in the same order as
	tiny-imagenet-200/wnids.txt. Class text descriptions are from
	tiny-imagenet-200/words.txt
	Returns:
		tuple of dicts
		label_dict:
			keys = synset (e.g. "n01944390")
			values = class integer {0 .. 199}
		class_desc:
			keys = class integer {0 .. 199}
			values = text description from words.txt
	"""
	label_dict, class_description = {}, {}
	with open('datasets/tiny-imagenet-200/wnids.txt', 'r') as f:
		for i, line in enumerate(f.readlines()):
			synset = line[:-1]  # remove \n
			label_dict[synset] = i
	with open('datasets/tiny-imagenet-200/words.txt', 'r') as f:
		for i, line in enumerate(f.readlines()):
			synset, desc = line.split('\t')
			desc = desc[:-1]  # remove \n
			if synset in label_dict:
				class_description[label_dict[synset]] = desc

	return label_dict, class_description

def read_image(filename_q, mode):
	"""Load next jpeg file from filename / label queue
	Randomly applies distortions if mode == 'train' (including a
	random crop to [56, 56, 3]). Standardizes all images.
	Args:
		filename_q: Queue with 2 columns: filename string and label string.
		filename string is relative path to jpeg file. label string is text-
		formatted integer between '0' and '199'
		mode: 'train' or 'val'
	Returns:
		[img, label]:
		img = tf.uint8 tensor [height, width, channels]  (see tf.image.decode.jpeg())
		label = tf.unit8 target class label: {0 .. 199}
	"""
	item = filename_q.dequeue()
	filename = item[0]
	label = item[1]
	file = tf.read_file(filename)
	img = tf.image.decode_jpeg(file, channels=3)
	# image distortions: left/right, random hue, random color saturation
	# if mode == 'train':
	# 	img = tf.random_crop(img, np.array([56, 56, 3]))
	# 	img = tf.image.random_flip_left_right(img)
	# 	# val accuracy improved without random hue
	# 	# img = tf.image.random_hue(img, 0.05)
	# 	img = tf.image.random_saturation(img, 0.5, 2.0)
	# else:
	# 	img = tf.image.crop_to_bounding_box(img, 4, 4, 56, 56)

	label = tf.string_to_number(label, tf.int32)
	label = tf.cast(label, tf.uint8)

	return [img, label]

def batch_q(mode, config):
	"""Return batch of images using filename Queue
	Args:
		mode: 'train' or 'val'
		config: training configuration object
	Returns:
		imgs: tf.uint8 tensor [batch_size, height, width, channels]
		labels: tf.uint8 tensor [batch_size,]
	"""
	filenames_labels = load_filenames_labels(mode)
	random.shuffle(filenames_labels)
	print('First step of pipeline...')
	# filename_q_OLD = tf.train.input_producer(filenames_labels, num_epochs=config.num_epochs, shuffle=True)
	filename_q = tf.data.Dataset.from_tensor_slices(filenames_labels).shuffle(tf.shape(filenames_labels, out_type=tf.int64)[0]).repeat(config.num_epochs)
	print('Done.')
	print(type(filename_q))

	# WARNING:tensorflow:From data_tiny.py:123: input_producer (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
	# Instructions for updating:
	# Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensor_slices(input_tensor).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.


	# 2 read_image threads to keep batch_join queue full:
	print('About to read images in batches...')
	# return tf.train.batch_join([read_image(filename_q, mode) for i in range(1)],
	# 							config.batch_size, shapes=[(56, 56, 3), ()],
	# 							capacity=2048)
	return tf.data.Dataset.from_tensors([read_image(filename_q, mode) for i in range(2)]).repeat(config.num_epochs)
	print('Done.')
	# WARNING:tensorflow:From C:\Users\sara\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow_core\python\training\input.py:189: limit_epochs (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
	# Instructions for updating:
	# Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensors(tensor).repeat(num_epochs)`.

def keras_reading(mode):
	mode = 'val'
	label_dict, class_description = build_label_dicts()
	dir = 'datasets/tiny-imagenet-200/' + mode
	image_generator = ImageDataGenerator(rescale=1./255)
	if mode == 'train':
		# train_dir = 'datasets/tiny-imagenet-200/train'
		# train_image_generator = ImageDataGenerator(rescale=1./255)
		train_data_gen = image_generator.flow_from_directory(batch_size=32,
	   															directory=dir,
																shuffle=True,
																target_size=(64, 64),
																class_mode='categorical')
		# print(type(train_data_gen.labels)) # <class 'numpy.ndarray'>
		# print(type(train_data_gen.labels[0])) # <class 'numpy.int32'>
		# print((train_data_gen.labels)) # values from 0 to 199
		# print(len(train_data_gen.labels)) # 10 000
	elif mode == 'val':
		val_labels = np.zeros(10000, dtype=np.int32)
		val_data_gen = image_generator.flow_from_directory(batch_size=32,
	   															directory=dir,
																shuffle=False,
																target_size=(64, 64),
																class_mode='binary')
		with open('datasets/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
			img_cnt = 0
			for line in f.readlines():
				split_line = line.split('\t')
				filename = 'datasets/tiny-imagenet-200/val/images/' + split_line[0]
				label = str(label_dict[split_line[1]])
				val_labels[img_cnt] = np.int32(int(label))
				# print(val_labels[img_cnt])
				# print(type(val_labels[img_cnt]))
				img_cnt += 1


		# sample_training_images, _ = next(train_data_gen) # returns a batch from the dataset on form x_train, y_train (one-hot-encoded)
		# sample_val_images = next(val_data_gen)
		# print(sample_val_images.filenames)
		print(type(val_data_gen))
		print('*********************')
		# print(val_data_gen.filepaths) # see here that the samples of val_data_gen are in order of filenames
		print(type(val_data_gen.labels)) # <class 'numpy.ndarray'>
		print(type(val_data_gen.labels[0])) # <class 'numpy.int32'>
		val_data_gen.classes = val_labels
		print(type(val_data_gen.labels)) # <class 'numpy.ndarray'>
		print(type(val_data_gen.labels[0])) # <class 'numpy.int32'>
		print((val_data_gen.labels)) # <class 'numpy.ndarray'>
		print((val_data_gen.labels[0])) # <class 'numpy.int32'>
		print(len(val_data_gen.labels)) # 10 000
		images, labels = next(val_data_gen)
		print(val_data_gen.classes)
		print(labels[:5])
		for x in labels[:5]:
			print(x)
			print(class_description[x])
		plotImages(images[:5])



	# plotImages(sample_training_images[:5])


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
	fig, axes = plt.subplots(1, 5, figsize=(20,20))
	axes = axes.flatten()
	for img, ax in zip( images_arr, axes):
		ax.imshow(img)
		ax.axis('off')
	plt.tight_layout()
	plt.show()

def main():

	batch_size = 128
	epochs = 50
	IMG_HEIGHT = 64
	IMG_WIDTH = 64
	# build_label_dicts() # OK!
	# load_filenames_labels('train') # OK!
	keras_reading('val')

	# mode = 'train'
	# my_config = config()
	# filename_q = batch_q(mode, my_config)
	# [img, label] = read_image(filename_q, mode)


if __name__ == '__main__':
	main()
