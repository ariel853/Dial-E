import pika
import json

connection_parameters = pika.ConnectionParameters('localhost')

connection = pika.BlockingConnection(connection_parameters)

channel = connection.channel()

channel.queue_declare(queue='letterbox')

message = {
    "from": "sender@example.com",
    "to": "receiver@example.com",
    "type": "notification",
    "content": "Hello, this is a custom RabbitMQ message!"
}
message_body = json.dumps(message)

channel.basic_publish(exchange='', routing_key='letterbox', body=message_body)

print(f"sent message: {message}")

connection.close()