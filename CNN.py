import pickle as cPickle
import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, Flatten, Activation
from keras import backend as k
import csv
import Helper as helper

#============ Multi-GPU ==========
#import multi_gpu
from tensorflow.keras.utils import multi_gpu_model

def baseline_model(strategy, tam_imagen, dropout, optimizer, activation, convolutional_layer_1, convolutional_layer_2, pooling_layer_1, pooling_layer_2):
	"""#Crear el modelo
	model = Sequential()
	#Primera capa entrada
	model.add(convolutional_layer_1)
	#model.add(activation)
	model.add(pooling_layer_1)
	#Segunda capa, intermedia
	model.add(convolutional_layer_2)
	#model.add(activation)
	model.add(pooling_layer_2)
	#ANN fully connected
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(rate=dropout))
	model.add(Dense(10, activation='softmax'))
	#============ Multi-GPU ============
	model = multi_gpu_model(model,gpus=2)
	#===================================
	#Compilar modelo
	model.compile(loss=keras.losses.categorical_crossentropy,
    			optimizer=keras.optimizers.Adadelta(),
    			metrics=['accuracy'])"""
	with strategy.scope():
		model = Sequential()
		model.add(convolutional_layer_1)
		model.add(pooling_layer_1)
		model.add(convolutional_layer_2)
		model.add(pooling_layer_2)
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(rate=dropout))
		model.add(Dense(10, activation='softmax'))

		model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=optimizer,
				metrics=['accuracy'])
	#strategy = tf.distribute.MirroredStrategy()
	#print('number of devices: {}'.format(strategy.num_replicas_in_sync))
	return model

def fitModel(datos_imagenes_entrenamiento, datos_target_entrenamiento, tam_imagen, epochs, dropout, optimizer, activation, convolutional_layer_1, convolutional_layer_2, pooling_layer_1, pooling_layer_2):
	#batch_size_per_replica = batch_size / strategy.num_replicas_in_sync
	#N gpus
	#strategy = tf.distribute.MirroredStrategy()
	#2 gpus
	strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0','/gpu:1'])
	#1 gpu
	#strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
	batch = 1000
	batch_size = batch * strategy.num_replicas_in_sync 
	#batch_size = 1000 
	buffer_size = 10000
	tamData = len(datos_imagenes_entrenamiento)
	train_dataset = tf.data.Dataset.from_tensor_slices((datos_imagenes_entrenamiento,datos_target_entrenamiento))
	train_dataset = train_dataset.shuffle(buffer_size).repeat().batch(batch_size)
	#Construyedo el modelo
	model = baseline_model(strategy, tam_imagen, dropout, optimizer, activation, convolutional_layer_1, convolutional_layer_2, pooling_layer_1, pooling_layer_2)
	#Empieza el entrenamiento con fit.
	print('Empezo el entrenamiento:')
	#model.fit(datos_imagenes_entrenamiento, datos_target_entrenamiento, batch_size=batch_size, epochs=epochs, verbose=1)
	model.fit(train_dataset, epochs=epochs,steps_per_epoch=tamData/batch, verbose=1)
	print('Finalizo el entrenamiento')
	return model


def experimento(serie, epochs, learning_rate, training_rate, optimizer, activation, filter_size, stride, padding, pool, dropout, numero_paso):
	#Mejor tamaño de imagen encontrado en experimento de tamaños de imagenes
	tam_imagen = 28
	#Mejor valor dropout encontrado en experimentos, pasado por el genético
	#valor_dropout = 0.4
	#Los datos, shuffled y split entre los conjuntos de entrenamiento y prueba
	num_clases = 10
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

	k.clear_session()
	if k.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, tam_imagen, tam_imagen)
		x_test = x_test.reshape(x_test.shape[0], 1, tam_imagen, tam_imagen)
		input_shape = (1, tam_imagen, tam_imagen)
		form = 'channels_first'
	else:
		x_train = x_train.reshape(x_train.shape[0], tam_imagen, tam_imagen, 1)
		x_test = x_test.reshape(x_test.shape[0], tam_imagen, tam_imagen, 1)
		input_shape = (tam_imagen, tam_imagen, 1)
		form = 'channels_last'

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	print('x_train shape: ', x_train.shape)
	print(x_train.shape[0], 'train samples.')
	print(x_test.shape[0], 'test samples.')

	y_train = keras.utils.to_categorical(y_train, num_clases)
	y_test = keras.utils.to_categorical(y_test, num_clases)
	opt = helper.optimizerFactory(optimizer, learning_rate)
	act = helper.activationFactory(activation)

	type = '2d'
	filter_number = 64
	kernel_size = (filter_size, filter_size)
	padd = padding

	conv1 = helper.convolutionLayerFactory('2d_first', filter_number, kernel_size, input_shape, padding, activation, form)
	conv2 = helper.convolutionLayerFactory(type, filter_number, kernel_size, input_shape, padding, activation, form)
	pool1 = helper.poolingFactory(pool)
	pool2 = helper.poolingFactory(pool)

	model = fitModel(x_train, y_train, tam_imagen, epochs, dropout, opt, act, conv1, conv2, pool1, pool2)
	score = model.evaluate(x_test, y_test, verbose=0)
	return score, model
