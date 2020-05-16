import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import random


''' read CIFAR10 dataset '''
def load_cifar10():
	valid_size = 5000
	n_classes = 10

	((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.cifar10.load_data()

	# split training data set into train and validation data sets
	x_valid, x_train = np.split(x_train, [valid_size], axis = 0)
	y_valid, y_train = np.split(y_train, [valid_size], axis = 0)

	# one hot encoding of labels - will we need this?
	# Y_train = one_hot(y_train, n_classes)
	# Y_valid = one_hot(y_valid, n_classes)
	# Y_test = one_hot(y_test, n_classes)

	# scale image data to 0-1
	x_train = x_train / 255
	x_valid = x_valid / 255
	x_test = x_test / 255

	return ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))

def augmentCifar(x_train):
	datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
	datagen.fit(x_train)
	return datagen

def one_hot(labels, n_classes):
	one_hot = np.zeros((labels.size, n_classes))
	for i in range(labels.size):
		label = labels[i][0]
		one_hot[i][label] = 1
	return one_hot


''' Only needed to run once, to store validation data in class subdirectories '''
def restructure_validation_data():
	print('Restructuring validation data...')
	dir = 'datasets/tiny-imagenet-200/val'
	img_dir = dir + '/images'
	img_dict = {}

	with open('datasets/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
		print('Val annotations file opened.')
		img_cnt = 0
		for line in f.readlines():
			split_line = line.split('\t')
			filename = 'datasets/tiny-imagenet-200/val/images/' + split_line[0]
			img_dict[split_line[0]] = split_line[1]

	print('Lenght of image dict: ', len(img_dict))
	for filename, label in img_dict.items():
		new_dir = (os.path.join(img_dir, label))
		if not os.path.exists(new_dir):
			os.makedirs(new_dir)
		if os.path.exists(os.path.join(img_dir, filename)):
			os.rename(os.path.join(img_dir, filename), os.path.join(new_dir, filename))

	print('Done restructuring validation data.')

''' read Tiny ImageNet dataset '''
def load_tiny(mode, batch_size):
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
	if mode not in ['train', 'val']:
		print("ERROR: mode must be 'train' or 'val'")
		return

	label_dict, class_description = build_label_dicts()

	if mode == 'train':
		dir = 'datasets/tiny-imagenet-200/' + mode
	elif mode == 'val':
		dir = 'datasets/tiny-imagenet-200/' + mode + '/images'

	augment_data = False
	if augment_data and mode == 'train': # we always only augment training data
		image_generator = ImageDataGenerator(rescale=1./255,
												width_shift_range=0.2,
												height_shift_range=0.2,
												horizontal_flip=True,
											)
	else:
		image_generator = ImageDataGenerator(rescale=1./255)

	data_gen = image_generator.flow_from_directory(batch_size=batch_size,
														directory=dir,
														shuffle=True,
														target_size=(64, 64),
														class_mode='categorical')

	return data_gen #, label_dict, class_description


def load_tiny_test(batch_size):
	''' Returns the Tiny ImageNet test images (withour labels) as a Keras DirectoryIterator '''
	dir = 'datasets/tiny-imagenet-200/test'
	image_generator = ImageDataGenerator(rescale=1./255)
	data_gen = image_generator.flow_from_directory(batch_size=batch_size,
														directory=dir,
														shuffle=True,
														target_size=(64, 64),
														class_mode=None)
	return data_gen


""" Function retrieved from: https://github.com/pat-coady/tiny_imagenet, Copyright (c) 2017 pat-coady """
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

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
	fig, axes = plt.subplots(1, 5, figsize=(20,20))
	axes = axes.flatten()
	for img, ax in zip( images_arr, axes):
		ax.imshow(img)
		ax.axis('off')
	plt.tight_layout()
	plt.show()

''' ARGS: dataset = str, data = imagegenerator (contains both image data and labels), n_classes = int, noise_rate = float, type = str '''
''' RETURN: updated imagegenerator '''
def add_noise_tiny(dataset, data, n_classes, noise_rate, type):
	clean_labels = data.labels
	noisy_labels = add_noise(dataset, clean_labels, n_classes, noise_rate, type)
	data.classes = noisy_labels
	new_labels = data.labels
	return data

''' add noise to data (of type symmetric or assymmetric) '''
def add_noise(dataset,labels,n_classes,noise_rate,type):
	if dataset not in ['cifar10', 'fashion_mnist']:
		print("ERROR: dataset must be 'cifar10' or 'fashion_mnist'")
		return

	# if dataset == 'imagenet': # make imagenet's 1D labels into 2D np.array
	# 	labels = np.array([labels]).transpose()

	noisy_labels=np.copy(labels)
	if type=='sym':
		for i in range(n_classes):
			ind_for_class = np.where(np.equal(labels,i))[0]
			np.random.shuffle(ind_for_class)
			ind_to_flip = ind_for_class[0:int(noise_rate*len(ind_for_class))]
			other_classes=list(range(n_classes))
			other_classes.remove(i)
			randoms = np.random.randint(0,n_classes-1,len(ind_to_flip))
			new_labels = [other_classes[i] for i in randoms]
			noisy_labels[ind_to_flip,:] = np.array(new_labels).reshape(len(ind_to_flip),1)
	else:
		if dataset == 'cifar10' :
			#TRUCK to AUTOMOBILE, BIRD to AIRPLANE, DEER to HORSE, CAT and DOG both ways
			flip ={2:0,9:1,4:7,3:5,5:3}
			for i in flip.keys():
				ind_for_class = np.where(np.equal(labels,i))[0]
				np.random.shuffle(ind_for_class)
				ind_to_flip = ind_for_class[0:int(noise_rate*len(ind_for_class))]
				noisy_labels[ind_to_flip,:]=flip[i]

		if dataset == 'fashion_mnist' :
			# SNEAKER to ANCLEBOOT, ANCLEBOOT to SNEAKER, PULLOVER to SHIRT, TSHIRT/TOP to PULLOVER, DRESS to COAT
			flip ={7:9,9:7,2:6,0:2,3:4}
			for i in flip.keys():
				ind_for_class = np.where(np.equal(labels,i))[0]
				np.random.shuffle(ind_for_class)
				ind_to_flip = ind_for_class[0:int(noise_rate*len(ind_for_class))]
				noisy_labels[ind_to_flip,:]=flip[i]

		# if dataset=='imagenet':
		# 	n_switched_classes = 40 # 40 switched classes randomly
		# 	flip={}
		# 	from_class=np.random.randint(0, n_classes, n_switched_classes)
		# 	to_class=np.random.randint(0, n_classes, n_switched_classes)
		# 	for i in range(n_switched_classes):
		# 		flip[from_class[i]]=to_class[i]
		# 	for i in flip.keys():
		# 		ind_for_class = np.where(np.equal(labels,i))[0]
		# 		np.random.shuffle(ind_for_class)
		# 		ind_to_flip = ind_for_class[0:int(noise_rate*len(ind_for_class))]
		# 		noisy_labels[ind_to_flip,:]=flip[i]

	# if dataset == 'imagenet':
	# 	noisy_labels = noisy_labels.transpose()[0]

	return noisy_labels

def main():
	# (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_cifar10()
	# noisy_labels = add_noise('cifar10',y_train,10,0.2,'sym')

	restructure_validation_data() # only need to be run once

if __name__ == '__main__':
	main()
