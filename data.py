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
def load_tiny_imagenet():
	x_train, Y_train, CLASS_NAMES = load_train_data_keras()
	x_valid = load_valid_data_keras()
	# TODO: load test data

	# return ((x_train, y_train, Y_train), (x_valid, y_valid, Y_valid), (x_test, y_test, Y_test))


''' reading images using keras.preprocessing as described in tutorial: https://www.tensorflow.org/tutorials/load_data/images '''
def load_train_data_keras():
	data_dir = pathlib.Path('C:/Users/sara/github/kth/DD2424-Project/datasets/tiny-imagenet-200/train')

	image_count = len(list(data_dir.glob('**/*.JPEG')))
	CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

	image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # convert from uint8 to float32 in range [0,1]
	BATCH_SIZE = 100000 # all the available training samples = 100 000
	IMG_HEIGHT = 64
	IMG_WIDTH = 64
	STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

	train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
													batch_size=BATCH_SIZE,
													shuffle=True,
													target_size=(IMG_HEIGHT, IMG_WIDTH),
													classes = list(CLASS_NAMES)
													)

	image_batch, label_batch = next(train_data_gen)
	# show_batch(image_batch, label_batch, CLASS_NAMES)
	return image_batch, label_batch, CLASS_NAMES


def load_valid_data_keras():
	data_dir = pathlib.Path('C:/Users/sara/github/kth/DD2424-Project/datasets/tiny-imagenet-200/val')

	image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # convert from uint8 to float32 in range [0,1]
	BATCH_SIZE = 10000
	IMG_HEIGHT = 64
	IMG_WIDTH = 64

	data_gen = image_generator.flow_from_directory(directory=str(data_dir),
													batch_size=BATCH_SIZE,
													shuffle=False,
													target_size=(IMG_HEIGHT, IMG_WIDTH),
													class_mode=None,
													)

	image_batch = next(data_gen)
	# TODO - read the labels of the validation data from the val_annotations.txt file
	return image_batch


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
