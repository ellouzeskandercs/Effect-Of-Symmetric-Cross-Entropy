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

	# scale image data to 0-1
	x_train = x_train / 255
	x_valid = x_valid / 255
	x_test = x_test / 255

	return ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))


def augmentCifar(x_train):
	datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
	datagen.fit(x_train)
	return datagen


''' read fashion mnist dataset '''
def load_fashion_mnist():
	valid_size = 6000
	((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.fashion_mnist.load_data()

	# reshape the labels to have rank 2 (needed for the add_noise function)
	y_train=y_train.reshape(y_train.shape+(1,))
	y_test = y_test.reshape(y_test.shape+(1,))

	#split training data set into train and validation data sets
	x_valid, x_train = np.split(x_train, [valid_size], axis = 0)
	y_valid, y_train = np.split(y_train, [valid_size], axis = 0)

	# scale image data to 0-1
	x_train = x_train / 255
	x_valid = x_valid / 255
	x_test = x_test / 255

	# reshape the data to have rank 4 (needed for the data augmentation)
	x_train= x_train.reshape(x_train.shape+(1,))
	x_test= x_test.reshape(x_test.shape+(1,))
	x_valid= x_valid.reshape(x_valid.shape+(1,))

	return ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))


''' add noise to data (of type symmetric or assymmetric) '''
def add_noise(dataset,labels,n_classes,noise_rate,type):
	if dataset not in ['cifar10', 'fashion_mnist']:
		print("ERROR: dataset must be 'cifar10' or 'fashion_mnist'")
		return
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

	return noisy_labels

