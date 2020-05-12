from data import *
from models import *
import tensorflow as tf
import matplotlib as plt
import numpy as np

# dataset = 'cifar10'
dataset = 'imagenet'
small_dataset = True

batch_size = 32

# read the data
if dataset == 'cifar10':
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_cifar10()
    datagen=augmentCifar(x_train)
    labels=y_train
    n_classes=10

elif dataset == 'imagenet':
    train_data_gen, _, _ = load_tiny('train')
    validation_data_gen, _, _ = load_tiny('val')
    test_data_gen = load_tiny_test()
    labels = train_data_gen.labels
    n_classes=200

noise_rates_sym=[0.2]#,0.3,0.4]
noise_rates_asym=[0.2,0.4,0.6,0.8]

for noise_rate in noise_rates_sym: #change here for assym
    # TODO - add noise to the ImageNet data
	# y_train=add_noise(dataset,labels,n_classes,noise_rate,type='sym')
    if small_dataset and dataset == 'cifar10':
        x_train = x_train[:100,:,:,:]
        y_train = y_train[:100,:]
        labels = labels[:100,:]
        # x_test = x_test[:1000,:,:,:]
        # y_test = y_test[:1000,:]

	# train the model
    if dataset == 'cifar10':
        model = get_model()
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),loss='mse',metrics=['accuracy']) # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
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
        H = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / 120, epochs=120, validation_data=(x_test, y_test), callbacks=[loss_history, lrate, Metr])

    if dataset == 'imagenet':
        # TODO - set parameters for optimal training according to http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf
        model = get_model_imagenet()
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),loss='mse',metrics=['accuracy']) # TODO - update loss function to correct one
        loss_history = LossHistory()
        Metr = Metrics_imagenet(model, train_data_gen, test_data_gen, labels, 200)
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay_imagenet)
        H = model.fit(train_data_gen, steps_per_epoch=10, epochs=2, validation_data=validation_data_gen, validation_steps=10, callbacks=[loss_history,lrate, Metr])

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
    plt.savefig('./ConfidenceCE'+str(NoiseRate)+'.png')

    Accuracies = np.array(Metr.train_acc_class).transpose()
    for i in range(10):
         plt.plot(range(60),Accuracies[i,:],'--',label='class'+str(i))
    plt.plot(range(60),H.history['val_accuracy'][:],'-r',label='overall')
    plt.legend()
    noise_rate = 0
    plt.savefig('./AccuracyperclassCE'+str(NoiseRate)+'.png')

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
    plt.savefig('./PredictionDistrCE'+str(NoiseRate)+'.png')
