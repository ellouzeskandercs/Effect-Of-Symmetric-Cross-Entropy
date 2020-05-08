# -*- coding: utf-8 -*-
"""Projet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hPxtmcnOk7RPQeYkGoNYqFTmxSMfIP4U

#Import Libraires
"""

import tensorflow as tf
import numpy as np
import math
import keras.backend as K
import matplotlib.pyplot as plt
from google.colab import drive
import tensorflow.keras.backend as K

"""#Download Data"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train[:5000,:,:,:]
y_train = y_train[:5000,:]

"""#Define The model"""

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

"""#Define the metrics"""

def accperclass0(y_true,y_pred,c=0):
 #return(K.shape(K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred),'float32'),y_true),'int32'))))
  #return(K.sum(K.cast(K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),(1000,1)),y_true),'int32')))
  z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
  e = K.equal(y_true,c)
  return(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.maximum(1,K.sum(K.cast(e,'int32'))))
def accperclass1(y_true,y_pred,c=1):
 #return(K.shape(K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred),'float32'),y_true),'int32'))))
  #return(K.sum(K.cast(K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),(1000,1)),y_true),'int32')))
  z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
  e = K.equal(y_true,c)
  return(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.maximum(1,K.sum(K.cast(e,'int32'))))
def accperclass2(y_true,y_pred,c=2):
 #return(K.shape(K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred),'float32'),y_true),'int32'))))
  #return(K.sum(K.cast(K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),(1000,1)),y_true),'int32')))
  z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
  e = K.equal(y_true,c)
  return(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.maximum(1,K.sum(K.cast(e,'int32'))))
def accperclass3(y_true,y_pred,c=3):
 #return(K.shape(K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred),'float32'),y_true),'int32'))))
  #return(K.sum(K.cast(K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),(1000,1)),y_true),'int32')))
  z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
  e = K.equal(y_true,c)
  return(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.maximum(1,K.sum(K.cast(e,'int32'))))
def accperclass4(y_true,y_pred,c=4):
 #return(K.shape(K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred),'float32'),y_true),'int32'))))
  #return(K.sum(K.cast(K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),(1000,1)),y_true),'int32')))
  z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
  e = K.equal(y_true,c)
  return(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.maximum(1,K.sum(K.cast(e,'int32'))))
def accperclass5(y_true,y_pred,c=5):
 #return(K.shape(K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred),'float32'),y_true),'int32'))))
  #return(K.sum(K.cast(K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),(1000,1)),y_true),'int32')))
  z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
  e = K.equal(y_true,c)
  return(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.maximum(1,K.sum(K.cast(e,'int32'))))
def accperclass6(y_true,y_pred,c=6):
 #return(K.shape(K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred),'float32'),y_true),'int32'))))
  #return(K.sum(K.cast(K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),(1000,1)),y_true),'int32')))
  z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
  e = K.equal(y_true,c)
  return(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.maximum(1,K.sum(K.cast(e,'int32'))))
def accperclass7(y_true,y_pred,c=7):
 #return(K.shape(K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred),'float32'),y_true),'int32'))))
  #return(K.sum(K.cast(K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),(1000,1)),y_true),'int32')))
  z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
  e = K.equal(y_true,c)
  return(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.maximum(1,K.sum(K.cast(e,'int32'))))
def accperclass8(y_true,y_pred,c=8):
 #return(K.shape(K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred),'float32'),y_true),'int32'))))
  #return(K.sum(K.cast(K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),(1000,1)),y_true),'int32')))
  z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
  e = K.equal(y_true,c)
  return(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.maximum(1,K.sum(K.cast(e,'int32'))))
def accperclass9(y_true,y_pred,c=9):
 #return(K.shape(K.sum(K.cast(K.equal(K.cast(K.argmax(y_pred),'float32'),y_true),'int32'))))
  #return(K.sum(K.cast(K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),(1000,1)),y_true),'int32')))
  z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
  e = K.equal(y_true,c)
  return(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.maximum(1,K.sum(K.cast(e,'int32'))))

"""#Train the model"""

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

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[accperclass0, accperclass1, accperclass2, accperclass3, accperclass4, accperclass5, accperclass6, accperclass7,accperclass8,accperclass9] +['accuracy'])

loss_history = LossHistory()
lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
H = model.fit(x_train,y_train, epochs=120,  validation_data= (x_test, y_test),callbacks=[loss_history, lrate])
with open('./drive/My Drive/objs.pkl', 'wb+') as f:  
    pickle.dump(H,f)

"""#Plot The results"""

plt.plot(range(len(H.history['lr'])),H.history['lr'])
print(H.history['lr'])

print(len(H.history['accperclass1']))
plt.plot(range(120),H.history['val_accperclass1'])
plt.plot(range(120),H.history['val_accperclass2'])
plt.plot(range(120),H.history['val_accperclass3'])

"""#Introducing Noise"""

def introduceSnoise(y,n_classes,p):
  res = np.copy(y)
  for i in range(n_classes):
    ind = np.where(np.equal(y,i))[0]
    np.random.shuffle(ind)
    ind = ind[0:int(p*len(ind))]
    oclasses = list(range(n_classes))
    oclasses.remove(i)
    randoms = np.random.randint(0,n_classes-1,len(ind))
    newlabels = [oclasses[i] for i in randoms]
    res[ind,:] = np.array(newlabels).reshape(len(ind),1)
  return(res)

#yn = introduceSnoise(y_train,10,0.4)
#ind = np.where(y_train == 6)[0]
#for j in range(10):
#  print(j,np.sum(np.equal(yn[ind,:],j))/len(ind))
y_train =  introduceSnoise(y_train,10,0.4)

def introduceASnoise(y,dataset,p):
  if dataset == 'cifar10' :
    #TRUCK → AUTOMOBILE, BIRD → AIRPLANE, DEER → HORSE, CAT ↔ DOG
    flip ={2:0,9:1,4:7,3:5,5:3}
    res = np.copy(y)
    for i in flip.keys():
      ind = np.where(np.equal(y,i))[0]
      np.random.shuffle(ind)
      ind = ind[0:int(p*len(ind))]
      res[ind,:] = flip[i]
  return(res)

yn = introduceASnoise(y_train,'cifar10',0.4)
ind = np.where(y_train == 2)[0]
for j in range(10):
  print(j,np.sum(np.equal(yn[ind,:],j))/len(ind))

"""#Import Loss"""

def custom_loss(y_actual,y_pred,A=-6,alpha=0.1,beta=1):
    q = K.one_hot(K.cast(y_actual,'uint8'),10)
    #custom_loss = - alpha * K.sum(K.dot(y_pred,K.minimum(K.log(y_actual),A))) - beta * K.sum(K.dot(y_pred,K.minimum(K.log(y_actual),A)))
    #custom_loss =  - alpha * K.sum(K.dot(K.transpose(y_pred),K.minimum(K.log(y_actual),A))) - beta * K.sum(K.dot(K.transpose(y_actual),K.minimum(K.log(y_pred),A)))
    custom_loss =  - alpha * K.mean(K.batch_dot(q,K.maximum(K.log(y_pred+1e-15),A))) - beta * K.mean(K.batch_dot(K.maximum(K.log(q+1e-15),A),y_pred))
    return custom_loss

"""#Train with noise"""

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./Chacks',
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),
              loss=custom_loss,
              metrics=[accperclass0, accperclass1, accperclass2, accperclass3, accperclass4, accperclass5, accperclass6, accperclass7,accperclass8,accperclass9] +['accuracy'])

loss_history = LossHistory()
lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
H = model.fit(x_train,y_train, epochs=120,callbacks=[loss_history, lrate,cp_callback])
with open('./drive/My Drive/objs.pkl', 'wb+') as f:  
    pickle.dump(H,f)

"""#Various"""

y = model.predict(x_test)
type(y)

y_pred = K.constant(y)
y_true = K.constant(y_test)

#K.all(keras.backend.stack([x, y], axis=0), axis=0)
for i in range(10): 
  z = K.equal(K.reshape(K.cast(K.argmax(y_pred),'float32'),K.shape(y_true)),y_true)
  e = K.equal(y_true,i)
  print(K.sum(K.cast(K.all(K.stack([z, e], axis=0), axis=0),'int32'))/K.sum(K.cast(e,'int32')))

K.shape(y_pred)

L = ['accperclass' + str(i) for i in range(10)]
print(L)

y_train = y_train [0: 20,:]
x_train = x_train [0 : 20 ,:]
a = K.constant(y_train)
q = K.one_hot(K.cast(a,'uint8'),10)
print(q.shape)
print(q)
y = model.predict(x_train)
print(y.shape)
y_pred = K.constant(y)
print(K.batch_dot(y_pred,K.maximum(K.log(q),-6)))

y = model.predict(x_test)
y_pred = K.constant(y)
y_true = K.constant(y_test)
q = K.one_hot(K.cast(y_true,'uint8'),10)

#custom_loss =  - alpha * K.mean(K.batch_dot(q,K.maximum(K.log(y_pred),A))) - beta * K.mean(K.batch_dot(K.maximum(K.log(q),A),y_pred))
print(y_pred)

model.weights[2]

type(model.load_weights('./Chacks.index'))