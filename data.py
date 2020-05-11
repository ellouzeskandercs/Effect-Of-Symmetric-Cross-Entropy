import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE

''' read CIFAR10 dataset '''
def load_cifar10():
	valid_size = 5000
	n_classes = 10

	((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.cifar10.load_data()

	# split training data set into train and validation data sets
	x_valid, x_train = np.split(x_train, [valid_size], axis = 0)
	y_valid, y_train = np.split(y_train, [valid_size], axis = 0)

	# one hot encoding of labels - will we need this?
	Y_train = one_hot(y_train, n_classes)
	Y_valid = one_hot(y_valid, n_classes)
	Y_test = one_hot(y_test, n_classes)

	# scale image data to 0-1
	x_train = x_train / 255
	x_valid = x_valid / 255
	x_test = x_test / 255

	return ((x_train, y_train, Y_train), (x_valid, y_valid, Y_valid), (x_test, y_test, Y_test))


def one_hot(labels, n_classes):
	one_hot = np.zeros((labels.size, n_classes))
	for i in range(labels.size):
		label = labels[i][0]
		one_hot[i][label] = 1
	return one_hot


''' read Tiny ImageNet dataset '''
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


def load_tiny(mode):
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
	elif mode == 'val': # todo - this does not work atm
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
		print(type(data_gen.labels)) # <class 'numpy.ndarray'>
		print(type(data_gen.labels[0])) # <class 'numpy.int32'>
		data_gen.classes = val_labels
		print(type(data_gen.labels)) # <class 'numpy.ndarray'>
		print(type(data_gen.labels[0])) # <class 'numpy.int32'>
		print((data_gen.labels))
		print((data_gen.labels[0]))
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


def show_batch(image_batch, label_batch, CLASS_NAMES):
	plt.figure(figsize=(10,10))
	for n in range(25):
		ax = plt.subplot(5,5,n+1)
		plt.imshow(image_batch[n])
		plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
		plt.axis('off')
	plt.show()


''' add noise to data (of type symmetric or assymmetric) '''
def add_noise(dataset,labels,n_classes,noise_rate,type):
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
			noisy_labels[ind_to_flip,:] = np.array(newl_abels).reshape(len(ind_to_flip),1)
	else:
		if dataset == 'cifar10' :
			#TRUCK to AUTOMOBILE, BIRD to AIRPLANE, DEER to HORSE, CAT and DOG both ways
			flip ={2:0,9:1,4:7,3:5,5:3}
			for i in flip.keys():
				ind_for_class = np.where(np.equal(labels,i))[0]
				np.random.shuffle(ind_for_class)
				ind_to_flip = ind_for_class[0:int(noise_rate*len(ind_for_class))]
				noisy_labels[ind_to_flip,:]=flip[i]

		if dataset== 'imageNet':
			# what are the classes??
			pass

	return noisy_labels


load_tiny_imagenet()
(x_train, y_train, Y_train), (x_valid, y_valid, Y_valid), (x_test, y_test, Y_test) = load_cifar10()
