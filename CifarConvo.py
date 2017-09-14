from keras.datasets import cifar10
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

seed = 1337
epochs = 10
# fix random seed for reproducibility
np.random.seed(seed)
x_max = 255.0
x_min = 0.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#standarization                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           X_train = (X_train - x_min )/ (x_max - x_min)
X_test = (X_test - x_min)/ (x_max - x_min)
X_train = (X_train - x_min)/ (x_max - x_min)



# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print('ilosc klas ' + str(num_classes))



model = Sequential()
model.add(Convolution2D(32,3,3,activation='relu',input_shape=( 3, 32, 32),dim_ordering="th", W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(4,4),border_mode='valid'))
model.add(Convolution2D(32,3,3, activation='linear',dim_ordering="th"))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
"""epoch = 10
learning_rate=0.01
decay = learning_rate/epoch
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learning_rate, momentum=0.9, nesterov=True))
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
"""
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))