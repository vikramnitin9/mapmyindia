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

theano.config.device = 'cpu'

stride_rows = 20
stride_cols = 20

input_shape = (3, 60, 60)

file = open('mean_var.pkl','rb')
mean_image, var_image = cPickle.load(file)
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

model.load_weights('weights.h5')

directory = os.listdir('./test')

table = []

for k, fname in enumerate(directory):
	print("Processing image ", k, end='\r')

	im = Image.open('./test/'+fname)

	im_rows = im.size[1]
	im_cols = im.size[0]

	draw = ImageDraw.Draw(im)
	
	max_prob = 0
	max_coords = [0,0]

	for i in range(0, im_rows - input_shape[1] + 1, stride_rows):
		for j in range(0, im_cols - input_shape[2] + 1, stride_cols):
			im_crop = im.crop((j, i, j+input_shape[2], i+input_shape[1]))
			im_crop = np.asarray(im_crop, dtype='float32')
			im_crop = np.transpose(im_crop, (2,0,1))
			im_crop /= 255.0

			assert (np.shape(im_crop) == input_shape), np.shape(im_crop)

			im_crop -= mean_image
			im_crop /= np.sqrt(var_image)
			im_crop = np.reshape(im_crop, (1,)+input_shape)

			output = model.predict(im_crop)[0]
				
			if(output > max_prob):
				max_prob = output
				max_coords = [i,j]

	draw.rectangle((max_coords[1], max_coords[0], max_coords[1]+input_shape[2], max_coords[0]+input_shape[1]), outline=(76,255,0))
				
	im.save('./predicted/'+(str)(k)+'.png')

	table.append([fname+'_Top', max_coords[0]])
	table.append([fname+'_Left', max_coords[1]])
	table.append([fname+'_Width', 60])
	table.append([fname+'_Height', 60])

print()
print(np.shape(table))

cols = ['Img_Label','Val']
df = pd.DataFrame(table, columns=cols)
df.to_csv('./submission.csv', index=False)

print()