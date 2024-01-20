import torch
from transformers import BartTokenizer, BartForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import time

class CustomDatasetMultiClass(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",  # Pad to max_length
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label
        }

# Example data for multi-class classification
firsttext = "show"
secondtext = """show
show
show
show"""
thirdtext = """show
show
show
show
show
show
show
show
show"""
train_texts = [firsttext, secondtext, thirdtext]
train_labels = [0, 1, 2] 

# 0 = normal
# 1 = interesting
# 2 = potential danger

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

# Load the BART tokenizer and model for sequence classification
model_name = 'facebook/bart-large-cnn'
model_directory = 'multi_class_classification_model_run_'
tokenizer = BartTokenizer.from_pretrained(model_name)
#use this line at first run:
#model = BartForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Three classes
model = BartForSequenceClassification.from_pretrained(model_directory)

# Create the dataset and DataLoader for multi-class classification - Training Set
train_dataset = CustomDatasetMultiClass(train_texts, train_labels, tokenizer, max_length=8)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Create the dataset and DataLoader for multi-class classification - Validation Set
val_dataset = CustomDatasetMultiClass(val_texts, val_labels, tokenizer, max_length=8)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Fine-tune the model for multi-class classification with learning rate adjustment
initial_lr = 5e-5
total_runs = 100
total_epochs_per_run = 5
total_warmup_steps = 0.1 * total_epochs_per_run * len(train_dataloader)
total_steps = total_runs * total_epochs_per_run * len(train_dataloader)

optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=initial_lr, total_steps=total_steps)
for epoch in range(total_epochs_per_run):
    total_loss = 0.0
    for batch in train_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    average_loss = total_loss / len(train_dataloader)

    # Validate the model on the validation set
    with torch.no_grad():
        val_loss = 0.0
        for val_batch in val_dataloader:
            val_input_ids = val_batch['input_ids']
            val_attention_mask = val_batch['attention_mask']
            val_labels = val_batch['labels']
            val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
            val_loss += val_outputs.loss.item()

        val_average_loss = val_loss / len(val_dataloader)

    print(f'Epoch {epoch + 1}, Training Loss: {average_loss}, Validation Loss: {val_average_loss}')

    # Save the model
model.save_pretrained(f'multi_class_classification_model_run_')
tokenizer.save_pretrained(f'multi_class_classification_model_run_')
time.sleep(5)

print("Done.")
