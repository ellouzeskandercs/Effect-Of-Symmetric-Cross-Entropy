from data import load_cifar10
from models import get_model,Metrics,LossHistory,step_decay
import tensorflow as tf
import matplotlib as plt
import numpy as np
from data import add_noise

# here we can write the main code of the project

# read the data
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_cifar10()
datagen=augmentCifar(x_train)

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
    labels = labels[:100,:]
	# x_test = x_test[:1000,:,:,:]
	# y_test = y_test[:1000,:]

	# train the model
    model = get_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),s=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
    loss_history = LossHistory()
    Metr = Metrics(model, x_train, y_train, labels, x_test, y_test)
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    # according to source aboove and code on github I think the training/fit should be:
    '''model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size, epochs=10,
                        validation_data=(x_test, y_test),
                        callbacks==[loss_history, lrate, Metr]
                        )'''
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
    
    fig, ax = plt.subplots()
    index = np.arange(10)
    bar_width = 0.35
    opacity = 0.8    
    rects1 = plt.bar(index, Metr.Pred[0], bar_width, alpha=opacity, color='lightblue', label='Predicted - Epoch 50')
    rects1 = plt.bar(index, Metr.CorrPred[0], bar_width, alpha=opacity, color='cornflowerblue', label='Correct - Epoch 50')   
    rects2 = plt.bar(index + bar_width, Metr.Pred[1], bar_width, alpha=opacity, color='lightcoral', label='Predicted - Epoch 100')
    rects2 = plt.bar(index + bar_width, Metr.CorrPred[1], bar_width, alpha=opacity, color='purple', label='Correct - Epoch 100')   
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Confidence distribution')
    plt.legend()
    plt.savefig('./PredictionDistrCE'+'NoiseRate.png')

	# test the model
