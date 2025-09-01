import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
category_map = {
    'Bank/account scams': 'Bank scams',
    'Employment scams': 'Job scams',
    'Government/institution impersonation scams': 'Gov/Institution scams',
    'Other': 'Other',
    'Relationship/dating scams': 'Romance scams',
    'Technical support/e-commerce scams': 'Tech/E-commerce scams',
    'Transfer scams': 'Money transfer scams'
}

# -------------------------------

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item['title'] + " " + item['text']
                label = category_map[item['category']]
                data.append({'text': text, 'label': label})
            except Exception as e:
                print(f"Error parsing line: {e}")
    return pd.DataFrame(data)

df = load_data('')


print(df['label'].value_counts())


label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
num_labels = len(label_encoder.classes_)
print(dict(zip(label_encoder.classes_, range(num_labels))))


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label_encoded'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']  
)


# -------------------------------


model_name = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


# -------------------------------

class ScamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_dataset = ScamDataset(train_texts, train_labels, tokenizer)
test_dataset = ScamDataset(test_texts, test_labels, tokenizer)


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# -------------------------------

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"{device}")

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 5
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# -------------------------------

model.train()
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    total_train_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"{avg_train_loss:.4f}")


# -------------------------------

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)


all_preds_labels = label_encoder.inverse_transform(all_preds)
all_labels_labels = label_encoder.inverse_transform(all_labels)


print("\n (classification_report):")
print(classification_report(all_labels_labels, all_preds_labels))


# -------------------------------

model.save_pretrained('./scam_bert_model')
tokenizer.save_pretrained('./scam_bert_model')