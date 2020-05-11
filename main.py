from data import load_cifar10
from models import get_model,Metrics,LossHistory,step_decay
import tensorflow as tf
import matplotlib as plt
import numpy as np
from data import add_noise

# here we can write the main code of the project

# read the data
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_cifar10()

noise_rates_sym=[0.2]#,0.3,0.4]
noise_rates_asym=[0.2,0.4,0.6,0.8]
cifar_dataset=True

if cifar_dataset:
     dataset='cifar10'
     labels=y_train
     n_classes=10
else:
     dataset='imageNet'
     n_classes=200

for noise_rate in noise_rates_sym: #change here for assym
	#y_train=add_noise(dataset,labels,n_classes,noise_rate,type='sym')
    x_train = x_train[:100,:,:,:]
    y_train = y_train[:100,:]
	# x_test = x_test[:1000,:,:,:]
	# y_test = y_test[:1000,:]

	# train the model
    model = get_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),s=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
    loss_history = LossHistory()
    Metr = Metrics(model, x_train, y_train, y_train, x_test, y_test)
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    H = model.fit(x_train,y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[loss_history, lrate, Metr])
   
    confidence = np.asarray(Metr.confidence)
    print(confidence)
    print(confidence.shape)
    fig1 = plt.figure()
    plt.bar(np.arange(confidence.shape[0]) + 1, confidence, color='r')
    plt.bar(np.arange(confidence.shape[0]) + 1, confidence, color='b')
    plt.legend()
    fig1.show()
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    fig2 = plt.figure()
    plt.plot(confidence)
    plt.xticks(np.arange(confidence.shape[0]))
    fig2.show
    plt.show()
    
    Accuracies = np.array(Metr.train_acc_class).transpose()
    for i in range(10):
         plt.plot(range(60),Accuracies[i,:],'--',label='class'+str(i))
    plt.plot(range(60),H.history['val_accuracy'][:],'-r',label='overall')
    plt.legend()
    noise_rate = 0
    plt.savefig('./AccuracyperclassCE'+'NoiseRate.png')
	# test the model
