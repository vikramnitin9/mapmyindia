# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.optimizers import SGD, RMSprop, Adam
# from keras.utils import np_utils

# import matplotlib.pyplot as plt
from six.moves import cPickle
import numpy as np

from PIL import Image

import os

import pandas as pd

resized_shape = (60, 60)
input_shape = (3, 60, 60)

print("Reading csv...")

df = pd.read_csv('train.csv', header=0)

df = df[df>0].dropna()

X_train = []
Y_train = []

count0 = 0
count1 = 0

for i,row in df.iterrows():
	print("Processing image ", i, end='\r')

	im = Image.open('./train/'+row['Img_Name'])

	imshape = im.size # (width, height)

	top = (int)(row['Top'])
	left = (int)(row['Left'])
	width = (int)(row['Width'])
	height = (int)(row['Height'])

	height = min(height, imshape[1]-top)
	width = min(width, imshape[0]-left)

	# b = border
	width_b = (int)(width/8)
	height_b = (int)(height/8)



	for j in range(-width_b, width_b, 3):
		for k in range(-height_b, height_b, 3):
			im_crop = im.crop((left + j, top + k, left + j + width, top + k + height))
			im_crop = im_crop.resize(resized_shape)
			im_crop.save('./data_full/1/'+(str)(count1)+'.jpg')
			count1 += 1

			im_crop = np.asarray(im_crop, dtype='float32')
			im_crop = np.transpose(im_crop, (2,0,1))

			assert (np.shape(im_crop) == input_shape)
			im_crop /= 255.0
			X_train.append(im_crop)
			Y_train.append(1)

	stride_rows = 150
	stride_cols = 100

	for j in range(0, im.size[1] - stride_cols, stride_cols):
		for k in range(0, im.size[0] - stride_rows, stride_rows):
			if (j in range(top-resized_shape[0], top+height) and k in range(left-resized_shape[1], left+width)):
				continue

			im_crop = im.crop((k, j, k+60, j+60))
			im_crop = im_crop.resize(resized_shape)
			# im_crop.save('./data/0/'+(str)(count0)+'.jpg')
			count0 += 1

			im_crop = np.asarray(im_crop, dtype='float32')
			im_crop = np.transpose(im_crop, (2,0,1))

			assert (np.shape(im_crop) == input_shape)
			im_crop /= 255.0
			X_train.append(im_crop)
			Y_train.append(0)

print()
print(count0, "Zeros and ", count1, "Ones")

print("Writing to pickle file...")

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)

p = np.random.permutation(len(X_train))
X_train = X_train[p]
Y_train = Y_train[p]

data = (X_train, Y_train)
file = open('train_data_full.pkl','wb')
cPickle.dump(data, file, cPickle.HIGHEST_PROTOCOL)
file.close()

file = open('train_data_full.pkl','rb')
(X_train, Y_train) = cPickle.load(file)
file.close()