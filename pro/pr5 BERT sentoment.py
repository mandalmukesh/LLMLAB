import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create and save a simple dataset
data = {
    'text': [
        'I love this product!',
        'This is the worst experience I have ever had.',
        'Absolutely fantastic! Highly recommend.',
        'Not what I expected, very disappointing.',
        'I am very satisfied with my purchase.',
        'Terrible service, will not return.',
        'Great quality, will buy again!',
        'I hate this, it broke after one use.'
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}
df = pd.DataFrame(data)
df.to_csv('sentiment_dataset.csv', index=False)

# Step 2: Load and preprocess the dataset
data = pd.read_csv('sentiment_dataset.csv')
data['text'] = data['text'].str.lower()
data['label'] = data['label'].map({'positive': 1, 'negative': 0})

# Step 3: Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(data['text'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)
labels = torch.tensor(data['label'].values)

# Step 4: Create DataLoader
train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs['input_ids'], labels, test_size=0.1, random_state=42)
train_masks, val_masks = train_test_split(inputs['attention_mask'], test_size=0.1, random_state=42)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=2)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=2)

# Step 5: Model Initialization
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 6: Training Loop
for epoch in range(4):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Training loss: {total_loss / len(train_dataloader):.4f}')

# Step 7: Evaluation
model.eval()
predictions, true_labels = [], []
for batch in val_dataloader:
    b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]
    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_attention_mask)
    logits = outputs.logits
    predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    true_labels.append(b_labels.cpu().numpy())

predictions = np.concatenate(predictions)
true_labels = np.concatenate(true_labels)
accuracy = accuracy_score(true_labels, predictions)
print(f'Validation Accuracy: {accuracy:.4f}')

# Step 8: Inference Function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return 'positive' if prediction == 1 else 'negative'

# Example usage
sample_text = "I love this product!"
print(f'Sentiment: {predict_sentiment(sample_text)}')