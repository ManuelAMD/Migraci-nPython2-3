from __future__ import print_function
import pika
import time
import json
import numpy as np
import time
import CNN as CNN
import os
import Helper as helper

class Solution(object):
	#Solución dada por el problema, esta compuesta de un valor binario y su valor fitness
	def __init__(self, dict):
		vars(self).update(dict)

	def calculate_fitness(self, fitness_function):
		scores = fitness_function(self.value)
		print(scores)
		self.fitness = float(scores[1])
		self.loss = float(scores[0])
		print("Calculando fitness")

def x(ch, method, properties, body):
	global cont
	global individuo1
	global individuo2

	cont = cont + 1

	f = lambda x: fitness(x)
	individuo = json.loads(body, object_hook=Solution)
	individuo.calculate_fitness(f)
	ch.basic_ack(delivery_tag=method.delivery_tag)
	#Credentials...
	connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', credentials=pika.PlainCredentials('guest','guest')))
	channel = connection.channel()
	channel.queue_declare(queue='individuosEntrenados', durable=True)
	individuo_entrenado = json.dumps(individuo.__dict__)
	channel.basic_publish(exchange='',
						routing_key='individuosEntrenados',
						body=individuo_entrenado)
	print('[x] Done')

def fitness(x):
	global serie
	global mejor_error
	global mejor_modelo
	global mejor_configuracion
	global numero_paso
	global contador_solucion

	contador_solucion = contador_solucion + 1
	castigo = [1.000000, -9999999]

	#Numero 0 no es válido
	#epocas 20, 30, 40, 50, 60, 70, ..., 140
	#Máximo binario 111 = 7
	num_epochs = x[:3]
	num_epochs = helper.binarioADecimal(num_epochs)
	num_epochs = (num_epochs + 1)*20

	#Lr 0.0001, 0.0006, 0.0011, 0.0016, 0.0021, 0.0026, 0.0031
	#Máximo binario 111 = 7
	#learning rate = 3 posiciones
	learning_rate = x[3:6]
	learning_rate = helper.binarioADecimal(learning_rate)
	switcher = {
		0: 0.0001,
		1: 0.0006,
		2: 0.0011,
		3: 0.0016,
		4: 0.0021,
		5: 0.0026,
		6: 0.0031,
		7: 0.0036
	}
	learning_rate = switcher.get(learning_rate, 'No se encontro el valor de learning_rate')

	#Training rate 0.60, 0.70, 0.80, 0.90 %
	#Máximo binario 11 = 3
	#training rate = 2 posiciones
	training_rate = x[6:8]
	training_rate = helper.binarioADecimal(training_rate)
	switcher = {
		0: 0.70,
		1: 0.80,
		2: 0.90,
		3: 1
	}
	training_rate = switcher.get(training_rate, 'No se encontro el valor de training_rate')

	#0 no es válido
	#Optimizadores de entrenamiento: GradientDescentOptimizer, AdamOptimizer and RMSPropOptimizer
	#Máximo binario 11 = 3
	#optimizer = 2 posiciones
	optimizer = x[8:10]
	optimizer = helper.binarioADecimal(optimizer)
	if optimizer == 0:
		return castigo
	switcher = {
		1: 'SGD',
		2: 'ADAM',
		3: 'RMSprop'
	}
	optimizer = switcher.get(optimizer, 'No se encontro el valor del optimizador')

	#valor 0 o 3 no son válidos
	#Activación relu o elu
	#Máximo binario 1 = 1
	#activación = 1 posición
	activacion = x[10]
	activacion = helper.binarioADecimal(activacion)
	switcher = {
		0: 'relu',
		1: 'elu'
	}
	activacion = switcher.get(activacion, 'No se encontro el valor de activación')

	#0 no es válido
	#Tamaños del filtro 3, 4, 5, 6
	#Máximo binario 11 = 3
	#filter_size = 2 posiciones
	filter_size = x[11:13]
	filter_size = helper.binarioADecimal(filter_size)
	filter_size = filter_size + 3

	#0 no es válido
	#Stride: 1, 2, 3, 4
	#Máximo binario 11 = 3
	#stride = 2 posiciones
	stride = x[13:15]
	stride = helper.binarioADecimal(stride)
	stride = stride + 1

	#Padding valid, same
	#Máximo binario 1 = 1
	#padding = 1 posición
	padding = x[15]
	padding = helper.binarioADecimal(padding)
	switcher = {
		0: 'valid',
		1: 'same'
	}
	padding = switcher.get(padding, 'No se encontro el valor de padding')

	#Capa de pooling: max_pooling, avg_pooling
	#Máximo binario 1 = 1
	#pool = 1 posición
	pool = x[16]
	pool = helper.binarioADecimal(pool)
	switcher = {
		0: 'MaxPooling2D',
		1: 'AveragePooling2D'
	}
	pool = switcher.get(pool, 'No se encontro el valor de pool')
	#Valor de dropout 0.3, 0.4, 0.5, 0.6
	#Máximo binario 11 = 3
	#valor_dropout = 2 posiciones
	valor_dropout = x[17:]
	valor_dropout = helper.binarioADecimal(valor_dropout)
	switcher = {
		0: 0.2,
		1: 0.3,
		2: 0.4,
		3: 0.5
	}
	valor_dropout = switcher.get(valor_dropout,'No se encontro el valor de dropout')

	nombre_archivo = "Agua_3121"
	ruta_archivo = nombre_archivo+'.csv'
	serie = helper.leerArchivo(ruta_archivo)
	serie = helper.normalizarSerie(serie)
	numero_paso = 1

	hora = time.strftime('%H:%M:%S')
	fecha = time.strftime('%d/%m/%Y')
	#Llamada a la CNN
	score, model = CNN.experimento(serie, num_epochs, learning_rate, training_rate, optimizer, activacion, filter_size, stride, padding, pool, valor_dropout, numero_paso)
	model.save('MejorModeloPaso'+str(numero_paso)+'.h5')
	print('Score: ',score)
	configuracion = 'NumEpochs:',num_epochs,'-LearningRate:',learning_rate,'-TrainingRate:',training_rate,'-Optimizer:',optimizer,'-Activación:',activacion,'-FilterSize:',filter_size,'-Stride:',stride,'-Padding:',padding,'-Pool:',pool,'-ValorDropout',valor_dropout,'-NumPaso:',numero_paso
	print(configuracion)
	return score


"""if 'HOME' not in os.environ:
	raise AssertionError('BROKER HOST variable not set')

broker_host = os.environ['HOME']"""

global cont
global individuo1
global individuo2
global numero_paso
global serie
global mejor_error
global mejor_modelo
global mejor_configuracion
global contador_solucion
global op

op = False

cont = 0
contador_solucion = 0

individuo1 = None
individuo2 = None

nombre_archivo = 'Agua_3121'
ruta_archivo = nombre_archivo+'.csv'
serie = helper.leerArchivo(ruta_archivo)
serie = helper.normalizarSerie(serie)

numero_paso = 1

print(nombre_archivo)

#Credentials = pika.PlainCredentials('server','nombre')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost',credentials=pika.PlainCredentials('guest','guest')))
														#heartbeat_interval=65535, blocked_connection_timeout=65535))
channel = connection.channel()
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='individuos', on_message_callback=x)
channel.start_consuming()
