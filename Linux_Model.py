"""
 To Detect Anomalies in a Linux System Log file
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import re
import os
from sklearn.metrics import confusion_matrix


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model and tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

def preprocess_log(log):
    log = re.sub(r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}', 'DATETIME', log)
    log = re.sub(r'(\d+\.\d+\.\d+)', 'VERSION', log)
    log = re.sub(r'(\d+)', 'NUM', log)
    return log

def load_logs_from_file(filepath, max_lines=400000):
    logs, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file.readlines()[:max_lines]:
            if "[Successful log]" in line:
                label = 0
                log = line.replace("[Successful log]", "").strip()
            elif "[Anomaly]" in line:
                label = 1
                log = line.replace("[Anomaly]", "").strip()
            else:
                continue
            logs.append(log)
            labels.append(label)
    return logs, torch.tensor(labels, dtype=torch.long).to(device)

def get_bert_embeddings(sentences, batch_size=256):
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch = [preprocess_log(sentence) for sentence in batch]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embeddings)

class Classifier(nn.Module):
    def _init_(self, input_dim, num_classes=2):
        super(Classifier, self)._init_()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Load logs
file_path = "/content/drive/MyDrive/Linux_labeled_full.txt"
logs, labels = load_logs_from_file(file_path)

# Class imbalance
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels.cpu().numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Train/test split
train_logs, test_logs, train_labels, test_labels = train_test_split(
    logs, labels, test_size=0.3, random_state=42
)

# Embeddings
train_embeddings = get_bert_embeddings(train_logs)
test_embeddings = get_bert_embeddings(test_logs)
train_tensor = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
test_tensor = torch.tensor(test_embeddings, dtype=torch.float32).to(device)

# Dataloaders
batch_size = 128
train_dataset = TensorDataset(train_tensor, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
classifier = Classifier(input_dim=train_embeddings.shape[1]).to(device)

# Training setup
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(classifier.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

epochs = 50
best_f1 = 0
patience = 5
patience_counter = 0

for epoch in range(epochs):
    classifier.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = classifier(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Validation
    classifier.eval()
    with torch.no_grad():
        val_preds = classifier(test_tensor)
        val_labels = torch.argmax(val_preds, dim=1).cpu().numpy()
        f1 = f1_score(test_labels.cpu().numpy(), val_labels)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(classifier.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping...")
                break

# Load best model
classifier.load_state_dict(torch.load("best_model.pt"))
classifier.eval()

# Final predictions
predicted_labels = []
with torch.no_grad():
    for i in range(0, test_tensor.shape[0], 500):
        batch = test_tensor[i:i+500]
        predictions = classifier(batch)
        predicted_labels.append(torch.argmax(predictions, dim=1))
predicted_labels = torch.cat(predicted_labels).cpu().numpy()

# Metrics
y_true = test_labels.cpu().numpy()
accuracy = accuracy_score(y_true, predicted_labels)
precision = precision_score(y_true, predicted_labels, zero_division=1)
recall = recall_score(y_true, predicted_labels, zero_division=1)
f1 = f1_score(y_true, predicted_labels, zero_division=1)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save anomalies
anomaly_logs = [log for log, label in zip(test_logs, predicted_labels) if label == 1]
with open("/content/Linux_anomalies.txt", "w", encoding='utf-8') as f:
    for log in anomaly_logs:
        f.write(log + "\n")
print("Anomalies saved to /content/Linux_anomalies.txt")

