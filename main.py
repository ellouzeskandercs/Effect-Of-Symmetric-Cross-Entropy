from data import *
from models import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils import save_metrics

'''Set the dataset'''
dataset_type = 'fashion_mnist'
# dataset_type = 'cifar10'

'''In case of test, a small part of the dataset may be used '''
small_dataset = False

'''Set the type of noise'''
noise_type = 'sym'
#noise_type = 'asym'

'''Read the data and set params'''
if dataset_type == 'cifar10':
    n_classes=10
    n_epochs = 120
    batch_size = 128 # set lower if memory error occur, otherwise a higher batch_size will give more stable gradients, but a too high value can also result in being stuck in local minima
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_cifar10()
    datagen=augmentCifar(x_train)
    labels=y_train # maybe not needed?

elif dataset_type == 'fashion_mnist':
    n_classes=10
    n_epochs = 120
    batch_size = 128 # set lower if memory error occur, otherwise a higher batch_size will give more stable gradients, but a too high value can also result in being stuck in local minima
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_fashion_mnist()
    print(x_train.shape,y_train.shape)
    datagen=augmentCifar(x_train)
    labels=y_train # maybe not needed?

'''Set the noise rates '''
if noise_type == 'sym':
    noise_rates= [0.0,0.2,0.4] # The 0.0 rate corresponds to the clean data
else: # 'asym'
    noise_rates=[0.2,0.4]

for noise_rate in noise_rates:
    tf.keras.backend.clear_session()
    loss_type = 'CE'
    if small_dataset and dataset_type == 'cifar10':
        x_train = x_train[:10,:,:,:]
        y_train = y_train[:10,:]
        labels = labels[:10,:]
        x_test = x_test[:10,:,:,:]
        y_test = y_test[:10,:]

    ''' add noise to training data - will be used for both models'''
    if dataset_type == 'cifar10':
        y_train = add_noise(dataset_type,labels,n_classes,noise_rate,type=noise_type)
    if dataset_type == 'imagenet':
        train_data_gen = add_noise_tiny(dataset_type, train_data_gen, n_classes, noise_rate, noise_type)
    if dataset_type == 'fashion_mnist':
        y_train = add_noise(dataset_type,labels,n_classes,noise_rate,type=noise_type)

	''' train the model with CE loss'''
    print('(1) Training with CL on the ',dataset_type,' dataset and using a ',noise_type,' noise rate ',noise_rate)
    if dataset_type == 'cifar10':
        model = get_model()
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
        loss_history = LossHistory()
        Metr = Metrics(model, x_train, y_train, labels, x_test, y_test,10)
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
        H = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / batch_size, epochs=n_epochs, validation_data=(x_test, y_test), callbacks=[loss_history, lrate, Metr])

    if dataset_type == 'fashion_mnist':
        model = get_model()
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
        loss_history = LossHistory()
        Metr = Metrics(model, x_train, y_train, labels, x_test, y_test,10)
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25)         
        H = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / batch_size, epochs=n_epochs, validation_data=(x_test, y_test), callbacks=[loss_history, lrate, Metr,es])

    ''' Save the metrcis on txt files'''
    save_metrics(Metr, H, dataset_type, loss_type, noise_type, noise_rate)

    ''' Plot the results'''
    # plotting the prediction confidence
    confidence = np.asarray(Metr.confidence)
    for i in range(2,C.shape[1]):
        plt.plot(range(C.shape[0]),C[i,:],'-o',label = 'epoch '+ str([5,10,30,50,70,90,110][i-2]))
    plt.xlabel('Class')
    plt.ylabel('Confidence')
    filename_confidence = './Confidence_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.png'
    plt.savefig(filename_confidence)
    plt.close()

    # plotting the Accuracies
    Accuracies = np.array(Metr.train_acc_class).transpose()
    for i in range(n_classes):
         plt.plot(range(n_epochs),Accuracies[i,:],'--',label='class'+str(i))
    plt.plot(range(n_epochs),Metr.acc,'-r',label='overall')
    plt.legend()
    filename_accperclass = './AccuracyPerClass_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.png'
    plt.title('Accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Class-wise test accuracy')
    plt.savefig(filename_accperclass)
    plt.close()

    # plot the bar diagram (prediction distribution)
    fig, ax = plt.subplots()
    index = np.arange(n_classes)
    bar_width = 0.35
    opacity = 0.8
    try:
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
    except IndexError as error:
        print(error)


    # Save model
    filename_model = './Model_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.txt'
    model.save(filename_model)

    tf.keras.backend.clear_session()

    ''' train the model with SL loss'''
    loss_type = 'SL'
    print('(2) Training with SL and ',dataset_type,' dataset and using a ',noise_type,' noise rate ',noise_rate)
    if dataset_type == 'cifar10':
        model = get_model()
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),loss=symmetric_cross_entropy)
        loss_history = LossHistory()
        Metr = Metrics(model, x_train, y_train, labels, x_test, y_test,10)
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
        H = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / batch_size, epochs=n_epochs, validation_data=(x_test, y_test), callbacks=[loss_history, lrate, Metr])

    if dataset_type == 'fashion_mnist':
        model = get_model()
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),loss=symmetric_cross_entropy)
        loss_history = LossHistory()
        Metr = Metrics(model, x_train, y_train, labels, x_test, y_test,10)
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25) 
        H = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / batch_size, epochs=n_epochs, validation_data=(x_test, y_test), callbacks=[loss_history, lrate, Metr,es])

    ''' Save the metrcis on txt files'''
    save_metrics(Metr, H, dataset_type, loss_type, noise_type, noise_rate)

    ''' Plot the results'''
    # plotting the prediction confidence
    confidence = np.asarray(Metr.confidence)
    for i in range(2,C.shape[1]):
        plt.plot(range(C.shape[0]),C[i,:],'-o',label = 'epoch '+ str([5,10,30,50,70,90,110][i-2]))
    plt.xlabel('Class')
    plt.ylabel('Confidence')
    filename_confidence = './Confidence_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.png'
    plt.savefig(filename_confidence)
    plt.close()

    # plotting the Accuracies
    Accuracies = np.array(Metr.train_acc_class).transpose()
    for i in range(n_classes):
         plt.plot(range(n_epochs),Accuracies[i,:],'--',label='class'+str(i))
    plt.plot(range(n_epochs),Metr.acc,'-r',label='overall')
    plt.legend()
    filename_accperclass = './AccuracyPerClass_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.png'
    plt.title('Accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Class-wise test accuracy')
    plt.savefig(filename_accperclass)
    plt.close()

    # plot the bar diagram (prediction distribution)
    fig, ax = plt.subplots()
    index = np.arange(n_classes)
    bar_width = 0.35
    opacity = 0.8
    try:
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
    except IndexError as error:
        print(error)

    # Save model
    filename_model = './Model_' + str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + str(noise_rate) + '.txt'
    model.save(filename_model)
