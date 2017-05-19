from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

import matplotlib.pyplot as plt
from six.moves import cPickle
import numpy as np

from PIL import Image

import os

def train_val_split(X, Y, frac=1.0):
	num_train_samples = X.shape[0]

	X_train = X[0 : (int)(frac*num_train_samples)]
	Y_train = Y[0 : (int)(frac*num_train_samples)]
	X_val = X[(int)(frac*num_train_samples) :]
	Y_val = Y[(int)(frac*num_train_samples) :]

	return (X_train, Y_train),(X_val, Y_val)

resized_shape = (60, 60)
input_shape = (3, 60, 60)

print("Loading data...")

file = open('train_data_full.pkl','rb')
(X_train, Y_train) = cPickle.load(file)
file.close()

print("Loaded!")

(X_train, Y_train), (X_val, Y_val) = train_val_split(X_train, Y_train, 0.9)

# Subtracting mean intensity over training set for each pixel
mean_image = np.zeros(input_shape,dtype='float32')
var_image = np.zeros(input_shape, dtype='float32')

for i in range(0, X_train.shape[0]):
    mean_image += (X_train[i]/X_train.shape[0])

for i in range(0, X_train.shape[0]):
    var_image += ((X_train[i]-mean_image)**2)/X_train.shape[0]

for i in range(0, X_train.shape[0]):
    X_train[i] -= mean_image
    X_train[i] /= np.sqrt(var_image)

for i in range(0, X_val.shape[0]):
    X_val[i] -= mean_image
    X_val[i] /= np.sqrt(var_image)

file = open('mean_var_full.pkl','wb')
cPickle.dump((mean_image, var_image), file, cPickle.HIGHEST_PROTOCOL)
file.close()

print("Compiling model...")

model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Convolution2D(64, 3, 3, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

datagen = ImageDataGenerator(
			# zoom_range=0.2,
			horizontal_flip=True)

datagen.fit(X_train)
train_generator = datagen.flow(X_train, Y_train, batch_size=40)

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('./weights/{epoch:02d}-{val_loss:.4f}.h5', save_weights_only=True)

# if(os.path.isfile('weights.h5')):
# 	print("Loading weights...")
# 	model.load_weights('weights.h5')
# 	print("Loaded!")

print("Getting ready to train...")

model.fit_generator(train_generator,
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=20,
                    validation_data=(X_val, Y_val), callbacks=[checkpoint])

# model.fit(X_train, Y_train, batch_size=40, nb_epoch=15, validation_split=0.1)
model.save_weights('weights.h5')

