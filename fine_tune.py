from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

import matplotlib.pyplot as plt
from six.moves import cPickle
import numpy as np

from PIL import Image, ImageDraw

import os

import pandas as pd

stride_rows = 20
stride_cols = 20

resized_shape = (60, 60)
input_shape = (3, 60, 60)

file = open('mean_var_full.pkl','rb')
mean_image_full, var_image_full = cPickle.load(file)
file.close()

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

model_full = model
model_full.load_weights('weights_full.h5')

print("Reading csv...")

df = pd.read_csv('submission_100.csv', header=0)

for k in range(0, df['Val'].size, 4):
	print((int)(k/4))
	fname = df['Img_Label'][k][:-4]
	im = Image.open('./test/'+ fname)

	im_rows = im.size[1]
	im_cols = im.size[0]

	draw = ImageDraw.Draw(im)
	
	max_coords = [0,0]
	max_coords[0] = df['Val'][k]
	max_coords[1] = df['Val'][k+1]

	# Fine tuning

	max_prob = 0
	new_max_coords = [0,0]
	max_size = 0

	for i in range(max_coords[0]-50, max_coords[0]+50, 10):
		for j in range(max_coords[1]-50, max_coords[1]+50, 10):
			for size in (15, 30, 40, 50, 60, 75):
				im_crop = im.crop((j, i, j+size, i+size))
				im_crop = im_crop.resize(resized_shape)
				im_crop = np.asarray(im_crop, dtype='float32')
				im_crop = np.transpose(im_crop, (2,0,1))
				im_crop /= 255.0

				assert (np.shape(im_crop) == input_shape), np.shape(im_crop)

				im_crop -= mean_image_full
				im_crop /= np.sqrt(var_image_full)
				im_crop = np.reshape(im_crop, (1,)+input_shape)

				output = model_full.predict(im_crop)[0]
				
				if(output > max_prob):
					max_prob = output
					new_max_coords = [i,j]
					max_size = size


	draw.rectangle((new_max_coords[1], new_max_coords[0], new_max_coords[1]+max_size, new_max_coords[0]+max_size), outline=(76,255,0))
				
	im.save('./predicted/'+(str)((int)(k/4))+'.png')

	df['Val'][k] = new_max_coords[0]
	df['Val'][k+1] = new_max_coords[1]
	df['Val'][k+2] = max_size
	df['Val'][k+3] = max_size

print()

df.to_csv('./submission1.csv', index=False)

print()