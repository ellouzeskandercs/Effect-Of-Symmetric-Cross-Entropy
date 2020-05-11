from data import load_cifar10
from models import get_model,Metrics,LossHistory,step_decay
import tensorflow as tf

# here we can write the main code of the project

# read the data
(x_train, y_train, Y_train), (x_valid, y_valid, Y_valid), (x_test, y_test, Y_test) = load_cifar10()
x_train = x_train[:1000,:,:,:]
y_train = y_train[:1000,:]
x_test = x_test[:1000,:,:,:]
y_test = y_test[:1000,:]

# train the model
model = get_model()
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
loss_history = LossHistory()
Metr = Metrics(model, x_train, y_train, y_train, x_test, y_test)
lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
H = model.fit(x_train,y_train, epochs=60,  validation_data= (x_test, y_test),callbacks=[loss_history, lrate,Metr])

# test the model
