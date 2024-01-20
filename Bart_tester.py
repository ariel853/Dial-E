import torch
from transformers import BartTokenizer, BartForSequenceClassification

# Load the trained model and tokenizer
model_name = 'facebook/bart-large-cnn'
model_directory = 'C:/Users/ariel/multi_class_classification_model_run_'  # Replace with the correct run directory
model = BartForSequenceClassification.from_pretrained(model_directory)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Define test sequences
test_sequences = ["show", """show
show
show
show""", 
"""show
show
show
show
show
show
show
show
show""", 
"""show 
show"""]

# Tokenize and prepare test data
test_inputs = tokenizer(test_sequences, add_special_tokens=True, padding="max_length", max_length=8, return_tensors="pt", truncation=True)

# Perform inference
with torch.no_grad():
    model.eval()
    outputs = model(**test_inputs)

# Get predicted probabilities for each class
predicted_probs = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()

# Define class labels
class_labels = ["normal", "interesting", "potential danger"]  # Update with your actual class labels

# Print results
for seq, probs in zip(test_sequences, predicted_probs):
    print(f"Sequence: {seq}, Predictions: {', '.join(f'{label} = {prob:.4f}' for label, prob in zip(class_labels, probs))}")
