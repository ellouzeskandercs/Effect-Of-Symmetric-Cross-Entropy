import glob
import re
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


""" Function rerieved from: https://github.com/pat-coady/tiny_imagenet, Copyright (c) 2017 pat-coady """
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


def keras_reading(mode):
	"""Generate data set based on mode.
	Args:
		mode: 'train' or 'val'
	Returns:
		data_gen:
			<class 'keras_preprocessing.image.directory_iterator.DirectoryIterator'>
			containing training or validation data rescaled to 0-1
		label_dict:
			keys = synset (e.g. "n01944390")
			values = class integer {0 .. 199}
		class_description:
			keys = class integer {0 .. 199}
			values = text description from words.txt
	"""
	label_dict, class_description = build_label_dicts()
	dir = 'datasets/tiny-imagenet-200/' + mode
	image_generator = ImageDataGenerator(rescale=1./255)

	if mode == 'train':
		data_gen = image_generator.flow_from_directory(batch_size=32,
	   															directory=dir,
																shuffle=True,
																target_size=(64, 64),
																class_mode='categorical')
	elif mode == 'val':
		data_gen = image_generator.flow_from_directory(batch_size=32,
	   															directory=dir,
																shuffle=False,
																target_size=(64, 64),
																class_mode='binary')
		# read the labels of the validation data from txt file
		val_labels = np.zeros(10000, dtype=np.int32)
		with open('datasets/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
			img_cnt = 0
			for line in f.readlines():
				split_line = line.split('\t')
				filename = 'datasets/tiny-imagenet-200/val/images/' + split_line[0]
				label = str(label_dict[split_line[1]])
				val_labels[img_cnt] = np.int32(int(label))
				img_cnt += 1

		print(type(data_gen)) # <class 'keras_preprocessing.image.directory_iterator.DirectoryIterator'>
		print('*********************')
		# print(val_data_gen.filepaths) # see here that the samples of val_data_gen are in order of filenames
		print(type(data_gen.labels)) # <class 'numpy.ndarray'>
		print(type(data_gen.labels[0])) # <class 'numpy.int32'>
		data_gen.classes = val_labels
		print(type(data_gen.labels)) # <class 'numpy.ndarray'>
		print(type(data_gen.labels[0])) # <class 'numpy.int32'>
		print((data_gen.labels)) # <class 'numpy.ndarray'>
		print((data_gen.labels[0])) # <class 'numpy.int32'>
		print(len(data_gen.labels)) # 10 000
		images, labels = next(data_gen)
		print(data_gen.classes)
		print(labels[:5])
		for x in labels[:5]:
			print(x)
			print(class_description[x])
		plotImages(images[:5])

	return data_gen, label_dict, class_description


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
	# build_label_dicts() # OK!
	keras_reading('val')


if __name__ == '__main__':
	main()
