# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:06:36 2020

@author: ASUS
"""
import tensorflow as tf
import numpy as np
import math
import tensorflow.keras.backend as K

def get_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv1'),
    tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=0.001),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv1'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),
    tf.keras.layers.Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv1'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(256, kernel_initializer="he_normal", kernel_regularizer= tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01), name='fc1'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu', name='lid'),
    tf.keras.layers.Dense(10, kernel_initializer="he_normal"),
    tf.keras.layers.Activation(tf.nn.softmax)
    ])
    return(model)
    
def accperclass(y_true,y_pred,c):
    z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
    e = K.equal(y_true,c)
    return(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.maximum(1,K.sum(K.cast(e,'int32'))))
    
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, model, X_train, y_train, y_train_clean, X_test, y_test):
        super(Metrics, self).__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_clean = y_train_clean
        self.X_test = X_test
        self.y_test = y_test
        self.n_class = y_train.shape[1]
        
        self.train_acc_class = []
        self.confidence = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch in [5, 10, 30, 50, 70, 90, 110]:
            predicted_prob = self.model.predict(self.X_train)
            self.confidence.append(np.mean(predicted_prob, axis=0))
        y = self.model.predict(self.X_test)
        y_pred = K.constant(y)
        y_true = K.constant(self.y_test)
        self.train_acc_class.append([K.eval(accperclass(y_true, y_pred, c=i)) for i in range(self.n_class)])
        print('AccperClass : ', self.train_acc_class[-1])

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 40
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
    
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))

def symmetric_cross_entropy(y_actual,y_pred,A=-6,alpha=0.1,beta=1):
    q = K.one_hot(K.cast(y_actual,'uint8'),10)
    custom_loss =  - alpha * K.mean(K.batch_dot(q,K.maximum(K.log(y_pred+1e-15),A))) - beta * K.mean(K.batch_dot(K.maximum(K.log(q+1e-15),A),y_pred))
    return custom_loss

