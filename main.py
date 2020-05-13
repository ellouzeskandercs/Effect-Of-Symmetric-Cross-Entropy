from data import *
from models import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils import save_metrics

dataset_type = 'imagenet'
dataset = ['imagenet', 'cifar10']
small_dataset = True
#noise_type = ['sym', 'asym']
noise_type = 'sym'
loss = ['CE', 'SL']

# read the data and set params
if dataset_type == 'cifar10':
    n_classes=10
    n_epochs = 120
    batch_size = 128 # set lower if memory error occur, otherwise a higher batch_size will give more stable gradients, but a too high value can also result in being stuck in local minima
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_cifar10()
    datagen=augmentCifar(x_train)
    labels=y_train # maybe not needed?

elif dataset_type == 'imagenet':
    n_classes=200
    n_epochs = 200 # not used if using early stopping
    batch_size = 128 # set lower if memory error occur, otherwise a higher batch_size will give more stable gradients, but a too high value can also result in being stuck in local minima
    steps_per_epoch = int(np.ceil(100000 / batch_size)) # 100 000 = number of training samples
    steps_per_val = int(np.ceil(10000 / batch_size))
    train_data_gen, _, _ = load_tiny('train', batch_size)
    validation_data_gen, _, _ = load_tiny('val', batch_size)
    test_data_gen = load_tiny_test(batch_size)
    labels = train_data_gen.labels


if noise_type == 'sym':
    noise_rates=[0.2]#,0.3,0.4]
else: # 'asym'
    noise_rates=[0.2,0.4,0.6,0.8]

for noise_rate in noise_rates:
    # TODO - add noise to the ImageNet data
	# y_train=add_noise(dataset,labels,n_classes,noise_rate,type=noise_type)
    loss_type = 'CE'
    if small_dataset and dataset_type == 'cifar10':
        x_train = x_train[:10,:,:,:]
        y_train = y_train[:10,:]
        labels = labels[:10,:]
        x_test = x_test[:10,:,:,:]
        y_test = y_test[:10,:]

	# train the model
    if dataset_type == 'cifar10':
        model = get_model()
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy']) # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss_history = LossHistory()
        Metr = Metrics(model, x_train, y_train, labels, x_test, y_test,10)
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
        # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
        # according to source aboove and code on github I think the training/fit should be:
        '''model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size, epochs=10,
                        validation_data=(x_test, y_test),
                        callbacks==[loss_history, lrate, Metr]
                        )'''
        H = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / batch_size, epochs=120, validation_data=(x_test, y_test), callbacks=[loss_history, lrate, Metr])

    if dataset_type == 'imagenet':
        # training according to http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf
        model = get_model_imagenet()
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=False),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
        loss_history = LossHistory()
        Metr = Metrics_imagenet(model, train_data_gen, test_data_gen, labels, n_classes)
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay_imagenet)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25) # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
        H = model.fit(train_data_gen, steps_per_epoch=steps_per_epoch, epochs=n_epochs, validation_data=validation_data_gen, validation_steps=steps_per_val, callbacks=[loss_history,lrate, Metr, es])


    # plotting the prediction confidence, TODO - plot only one specific class
    confidence = np.asarray(Metr.confidence)
    plt.plot(confidence)
    plt.xticks(np.arange(confidence.shape[0]))
    filename_confidence = './Confidence_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.png'
    plt.savefig(filename_confidence)
    plt.close()

    Accuracies = np.array(Metr.train_acc_class).transpose()
    for i in range(n_classes):
         plt.plot(range(n_epochs),Accuracies[i,:],'--',label='class'+str(i))
    plt.plot(range(n_epochs),H.history['val_accuracy'][:],'-r',label='overall')
    plt.legend()
    # noise_rate = 0
    filename_accperclass = './AccuracyPerClass_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.png'
    plt.savefig(filename_accperclass)
    plt.close()

    # plot the bar diagram (prediction distribution)
    fig, ax = plt.subplots()
    index = np.arange(n_classes)
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
    filename_PredictionDist = './PredictionDistr_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.png'
    plt.savefig(filename_PredictionDist)
    plt.close()

    # Save model
    filename_model = './Model_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.png'
    model.save(filename_model)

    loss_type = 'SL'
    if dataset_type == 'cifar10':
        model = get_model()
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),loss=symmetric_cross_entropy,metrics=['accuracy']) # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss_history = LossHistory()
        Metr = Metrics(model, x_train, y_train, labels, x_test, y_test,10)
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
        # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
        # according to source aboove and code on github I think the training/fit should be:
        '''model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size, epochs=10,
                        validation_data=(x_test, y_test),
                        callbacks==[loss_history, lrate, Metr]
                        )'''
        H = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / batch_size, epochs=120, validation_data=(x_test, y_test), callbacks=[loss_history, lrate, Metr])

    if dataset_type == 'imagenet':
        # TODO - set parameters for optimal training according to http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf
        model = get_model_imagenet()
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),loss=symmetric_cross_entropy,metrics=['accuracy']) # TODO - update loss function to correct one
        loss_history = LossHistory()
        Metr = Metrics_imagenet(model, train_data_gen, test_data_gen, labels, 200)
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay_imagenet)
        H = model.fit(train_data_gen, steps_per_epoch=10, epochs=2, validation_data=validation_data_gen, validation_steps=10, callbacks=[loss_history,lrate, Metr])

    # todo - save all data to files
    save_metrics(Metr, history, dataset_type, loss_type, noise_type, noise_rate):
    # todo - save the trained model

    # plotting the prediction confidence, TODO - plot only one specific class
    confidence = np.asarray(Metr.confidence)
    plt.plot(confidence)
    plt.xticks(np.arange(confidence.shape[0]))
    filename_confidence = './Confidence_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) +  '.png'
    plt.savefig(filename_confidence)
    plt.close()

    Accuracies = np.array(Metr.train_acc_class).transpose()
    for i in range(n_classes):
         plt.plot(range(n_epochs),Accuracies[i,:],'--',label='class'+str(i))
    plt.plot(range(n_epochs),H.history['val_accuracy'][:],'-r',label='overall')
    plt.legend()
    # noise_rate = 0
    filename_accperclass = './AccuracyPerClass_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.png'
    plt.savefig(filename_accperclass)
    plt.close()

    # plot the bar diagram (prediction distribution)
    fig, ax = plt.subplots()
    index = np.arange(n_classes)
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
    filename_PredictionDist = './PredictionDistr_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.png'
    plt.savefig(filename_PredictionDist)
    plt.close()

    # Save model
    filename_model = './Model_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.png'
    model.save(filename_model)
