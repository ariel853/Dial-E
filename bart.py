from transformers import pipeline
import time


classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

labels = ["ping command", "show command", "set command"]
hypothesis_template = """this sequence is {}."""
sequence = "ping 1.1.1.1"
sequence_arr = ["ping 1.1.1.1", "set ip 192.168.1.1", "show version"]
for x in sequence_arr:
    prediction = classifier(x, labels, hypothesis_template=hypothesis_template, multi_class=True)
    print(prediction)
