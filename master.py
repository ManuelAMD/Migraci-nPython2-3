from random import random
import Helper as helper
import time
import numpy as np
import pika
import json
import os
from ThreadRabbitMq import ThreadRabbitMq

class Solution(object):
	#Una solución para el problema dado, esta compuesta de un valor binario y su valor fitness
	def __init__(self, dict):
		vars(self).update(dict)

	def calculate_fitness(self, fitness_function):
		self.fitness = fitness_function(self.value)


def recieve_individuos(individuo):
	global stop
	global contador_individuos_entrenados
	global individuos_entrenado_uno
	global individuos_entrenado_dos

	individuo = json.loads(individuo, object_hook=Solution)

	print('Recibiendo individuo: ', individuo)
	#print('Individuo.value: ', individuo.value)
	#print('Recibiendo individuo: ', individuo.fitness)

	contador_individuos_entrenados = contador_individuos_entrenados + 1

	print(contador_individuos_entrenados)
	if contador_individuos_entrenados == 2:
		individuos_entrenado_dos = individuo
		stop = False
		contador_individuos_entrenados = 0
	else:
		individuos_entrenado_uno = individuo

def generate_candidate(vector):
	#Se generan una nueva solución candidata basada en el vector de probabilidad.
	value = ''
	for p in vector:
		value += "1" if random() < p else "0"

	dictionary = {
		'value': value,
		'fitness': 0
	}

	return Solution(dictionary)

def generate_vector(size):
	#Inicializa un vector de probabilidad con el tamaño dado
	return [0.5] * size

def compete(a, b):
	#Retorna una tupla con la solución ganadora
	if a.fitness > b.fitness:
		return a, b
	else:
		return b, a

def update_vector(vector, winner, loser, population_size):
	#Individuo 0111000101100001110
	for i in range(len(vector)):
		if winner[i] != loser[i]:
			if winner[i] == '1':
				vector[i] += 1.0/float(population_size)
			else:
				vector[i] -= 1.0/float(population_size)

def run(generations, size, population_size, fitness_function):
	global stop
	global vector
	global individuos_entrenado_uno
	global individuos_entrenado_dos
	#Probabilidad por cada bit de la solución sea 1
	vector = generate_vector(size)
	#[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
	best = None

	#Se detiene dado el numero de generaciones, pero se puede definir cualquier parametro para parar
	for i in range(generations):
		stop = True
		print('GENERACIÓN : ', i)
		#Se generan dos soluciones candidatas, es como la selección en un AG convencional.
		s1 = generate_candidate(vector)
		s2 = generate_candidate(vector)
		#Ejemplo generado: 0111000101100001110
		#print('Connecting to host ', broker_host,' on default port.')
		#connection = pika.BlockingConnection(pika.ConnectionParameters(host=broker_host))
		connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
		channel = connection.channel()

		channel.queue_declare(queue='individuos', durable=True)

		s1j = json.dumps(s1.__dict__) #forma de json {"value": "0110001111110111101", "fitness": 0}
		s2j = json.dumps(s2.__dict__)
		print(s1j)
		print(s2j)

		channel.basic_publish(exchange='',
							routing_key='individuos',
							body=s1j)
		channel.basic_publish(exchange='',
							routing_key='individuos',
							body=s2j)
		connection.close()
		while(stop):
		#	print("hoo")
			a = 1

		print("Inicia Competencia")
		winner, loser = compete(individuos_entrenado_uno, individuos_entrenado_dos)
		print("Finaliza Competencia")
		if best:
			if winner.fitness > best.fitness:
				best = winner
		else:
			best = winner
		print("Generación: %d mejor valor: %s mejor fitness: %f"%(i + 1, best.value, float(best.fitness)))
		#Actualiza el vector de probabilidades basado en el exito de cada bit
		update_vector(vector, winner.value, loser.value, population_size)
		#return self.mejorModelo

def callback(body):
	global stop
	print(body)
	stop = False

if __name__ == '__main__':

	"""print(os.environ)

	if 'BROKER_HOST' not in os.environ:
		raise AssertionError('BROKER HOST varible not set')

	broker_host = os.environ['broker_host']"""

	hora = time.strftime("%H-%M-%S")
	fecha = time.strftime("%d-%m-%Y")

	global numero_paso
	global serie
	global mejor_error
	global mejor_modelo
	global mejor_configuracion
	global stop
	global vector
	global individuos_entrenado_uno
	global individuos_entrenado_dos
	global contador_individuos_entrenados

	individuos_entrenado_uno = None
	individuos_entrenado_dos = None
	vector = None

	contador_individuos_entrenados = 0

	stop = False

	print("Lanzando escuchador de cuando los individuos ya han finalizado")
	td = ThreadRabbitMq('localhost','guest','guest','individuosEntrenados', recieve_individuos, durable=True)
	td.start()

	for numero_paso in range(1, 2):
		mejor_error = 99999999
		mejor_modelo = None
		mejor_configuracion = None

		f = lambda x: fitness(x)
		run(10, 19, 100, f)
		print('<<<<<<<<<<<<<<<< MEJOR CONFIGURACIÓN PASO ', numero_paso, " >>>>>>>>>>>>>>>>>>")

		print(mejor_configuracion)

		#nombre_iteracion = "paso"+str(numero_paso)
		#helper.guardarMSEContinuoExcel('Experimento_resultadosMSE'+fecha+'_'+hora+'.csv',nombre_iteracion, mejor_error, 'MSE_TEST', mejor_configuracion, 'MEJOR CONFIGURACIÓN')

	print('Inicio: ', fecha, ' ', hora, ' Termino: ', time.strftime('%H:%M:%S'), ' ', time.strftime('%d/%m/%Y'))