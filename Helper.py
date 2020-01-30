import csv
from collections import deque
from bisect import insort, bisect_left
from itertools import islice
import logging
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, AveragePooling2D
#from keras.layers.convolutional import *

def leerArchivo(ruta):
	serie = []
	with open(ruta, 'rt') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=' ',quotechar='|')
		for row in spamreader:
			dato = float(row[0])
			serie.append(dato)
	return serie

def normalizarSerie(serie):
	minimo = float(min(serie))
	maximo = float(max(serie))
	dividendo = float(maximo-minimo)
	print(dividendo)
	return map(lambda x: (x-minimo)/dividendo, serie)

def binarioADecimal(binario):
	binarioStr = ''.join(str(x) for x in binario)
	return int(binarioStr, 2)

def guardarMSEContinuoExcel(dirArchivoCSV, tag1, valor1, tagValor1, valor2, tagValor2):
	with open(dirArchivoCSV, 'a') as archivo_nuevo_csv:
		writer = csv.DictWriter(archivo_nuevo_csv, fieldnames=['iteracion', tagValor1, tagValor2])
		for i in range(0, len(valor1)):
			valor1 = str(valor1[i])
			valor1 = valor1.replace('[','')
			valor1 = valor1.replace(']','')
			writer.writerow({'iteracion': tag1, tagValor1: valor1, tagValor2: valor2})

def optimizerFactory(optimizer, learning_rate):
	if optimizer == 'SGD':
		return optimizers.SGD(lr=learning_rate)
	elif optimizer == 'ADAM':
		return optimizers.Adam(lr=learning_rate)
	elif optimizer == 'RMSprop':
		return optimizers.RMSprop(lr=learning_rate)
	else:
		logging.warning('Optimizer incorrecto!!')
		raise

def activationFactory(activation):
	if activation == 'relu':
		return Activation('relu')
	elif activation == 'elu':
		return Activation('elu')
	else:
		logging.warning('Activación incorrecta!!')
		raise

def convolutionLayerFactory(type, filter_number, kernel_size, input_shape, padding, activation, form):
	#print('FILTER: ', filter_number, '--kernel_size: ', kernel_size, '--Activation:',activation,'---Shape:',input_shape)
	if type == '2d_first':
		return Conv2D(filter_number, kernel_size, padding=padding, data_format=form, activation=activation, input_shape=input_shape)
	elif type == '2d':
		return Conv2D(filter_number, kernel_size, padding=padding, data_format=form, activation=activation)
	else:
		logging.warning('Tipo capa convolución incorrecto!!')
		raise

def poolingFactory(type):
	if type == 'MaxPooling2D':
		return MaxPooling2D(pool_size=(2,2))
	elif type == 'AveragePooling2D':
		return AveragePooling2D(pool_size=(2,2))
	else:
		logging.warning('Tipo pooling incorrecto!!')