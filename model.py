import os
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from keras.layers.core import Activation


class SRCNN(keras.Model):
	def __init__(self):
		self.input = Input(input_shape=(2160, 3840, 1))
		self.conv1 = Conv2D(64,9,padding='same')
		self.act1 = Activation('relu')
		self.conv2 = Conv2D(32,1,padding='same')
		self.act2 = Activation('relu')
		self.conv3 = Conv2D(1, padding='same')

		self.seq = Sequential(
			self.input,
			self.conv1,
			self.act1,
			self.act2,
			self.conv3
			)

	def call(self, input):
		return self.seq(input)
