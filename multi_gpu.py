from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Concatenate, Lambda
from tensorflow.keras import backend as k

session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=session_config)

def slice_batch(x, n_gpus, part):
	sh = k.shape(x)
	l = sh[0] // n_gpus
	if part == n_gpus - 1:
		return x[part*l:]
	return x[part*l:(part+1)*l]

def to_multi_gpu(model, n_gpus=2):
	print("XXXXXXXXXXXXXXXXXX",tf.device('/cpu:0'))
	if n_gpus == 1:
		return model

	with tf.device('/cpu:0'):
		x = Input(model.input_shape[1:])
	towers = []
	for g in range(n_gpus):
		with tf.device('/gpu:'+str(g)):
			slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':n_gpus, 'part':g})(x)
			towers.append(model(slice_g))
	with tf.device('/cpu:0'):
		merged = Concatenate(axis=0)(towers)
	return Model(inputs=[x], outputs=merged)
