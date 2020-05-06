import tensorflow as tf
import numpy as np
# import numpy as np
''' read CIFAR-10 dataset '''
def read_cifar10():
	((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.cifar10.load_data()
	valid_size = 5000
	n_classes = 10

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
	one_hot = np.zeros((n_classes, labels.size))
	for i in range(labels.size):
		label = labels[i][0]
		one_hot[label][i] = 1
	return one_hot

''' read Tiny ImageNet dataset '''


''' add noise to data '''



(x_train, y_train, Y_train), (x_valid, y_valid, Y_valid), (x_test, y_test, Y_test) = read_cifar10()
print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
