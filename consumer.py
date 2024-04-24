import pika
import json

def on_message_received(ch, method, properties, body):
    print(f"received new message: {body}")
    parser(body)


def parser(message1):
    message = json.loads(message1)
    message_from = message.get("from", "")
    message_to = message.get("to", "")
    message_type = message.get("type", "")
    message_content = message.get("content", "''")
    message_array = [message_from, message_to, message_type, message_content]

    print(message_array)


if __name__ == '__main__':

    connection_parameters = pika.ConnectionParameters('localhost')

    connection = pika.BlockingConnection(connection_parameters)

    channel = connection.channel()

    channel.queue_declare(queue='letterbox')

    channel.basic_consume(queue='letterbox', auto_ack=True,
        on_message_callback=on_message_received)

    print("Starting Consuming")

    channel.start_consuming()
