import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
connection.close()
