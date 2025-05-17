"""
 To Detect Anomalies in a BGL System Log file
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Use GPU if available, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

def preprocess_log(log):
    """
    Preprocess log messages by replacing timestamps, dates, times, and node identifiers with placeholders.
    """
    log = re.sub(r'\d{10}', 'TIMESTAMP', log)
    log = re.sub(r'\d{4}\.\d{2}\.\d{2}', 'DATE', log)
    log = re.sub(r'\d{2}:\d{2}:\d{2}\.\d+', 'TIME', log)
    log = re.sub(r'R\d+-M\d+-N\d+-C:J\d+-U\d+', 'NODE', log)
    return log

def load_logs_from_file(filepath, max_lines=400000):
    """
    Reads log file, extracts log messages, and labels them as normal (0) or anomaly (1).
    The first column in the log file determines the label:
    - "-" (dash) indicates normal logs (0)
    - Any other value indicates an anomaly (1)
    """
    logs, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file.readlines()[:max_lines]:
            parts = line.strip().split(" ", 1)
            if len(parts) > 1:
                label = 0 if parts[0] == "-" else 1
                labels.append(label)
                logs.append(parts[1])
    return logs, torch.tensor(labels, dtype=torch.long).to(device)

def get_bert_embeddings(sentences, batch_size=500):
    """
    Converts log messages into numerical embeddings using a pre-trained BERT model.
    """
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
    """
    A feed-forward neural network classifier for anomaly detection.
    """
    def __init__(self, input_dim, num_classes=2):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)

# Mount Google Drive to access log files (since the code was run through google colab)
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Load logs and labels from file
file_path = "/content/drive/MyDrive/BGL.log"
logs, labels = load_logs_from_file(file_path, max_lines=400000)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.array([0,1]), y=labels.cpu().numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Split data into training and testing sets (70% train, 30% test)
train_logs, test_logs, train_labels, test_labels = train_test_split(logs, labels, test_size=0.3, random_state=42)

# Convert logs into BERT embeddings
train_embeddings = get_bert_embeddings(train_logs, batch_size=500)
test_embeddings = get_bert_embeddings(test_logs, batch_size=500)

train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32).to(device)

classifier = Classifier(input_dim=train_embeddings.shape[1]).to(device)

# Train classifier on labeled data
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

epochs = 10  # Train the model for 10 epochs
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = classifier(train_embeddings_tensor)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Predict anomalies on test set
predicted_labels = []
with torch.no_grad():
    for i in range(0, test_embeddings_tensor.shape[0], 500):
        batch = test_embeddings_tensor[i:i+500]
        predictions = classifier(batch)
        predicted_labels.append(torch.argmax(predictions, dim=1))
predicted_labels = torch.cat(predicted_labels).cpu().numpy()

# Evaluate Model Performance
accuracy = accuracy_score(test_labels.cpu().numpy(), predicted_labels)
precision = precision_score(test_labels.cpu().numpy(), predicted_labels, zero_division=1)
recall = recall_score(test_labels.cpu().numpy(), predicted_labels, zero_division=1)
f1 = f1_score(test_labels.cpu().numpy(), predicted_labels, zero_division=1)

# Print model performance metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save anomaly logs to a separate file for further analysis
anomaly_logs = [log for log, label in zip(test_logs, predicted_labels) if label == 1]
with open("/content/anomalies.txt", "w", encoding='utf-8') as f:
    for log in anomaly_logs:
        f.write(log + "\n")

print("Anomalies saved to /content/anomalies.txt")
cm = confusion_matrix(test_labels.cpu().numpy(), predicted_labels)
print("\nConfusion Matrix:")
print("               Predicted: Normal    Predicted: Anomaly")
print(f"Actual: Normal       {cm[0][0]:<18} {cm[0][1]}")
print(f"Actual: Anomaly      {cm[1][0]:<18} {cm[1][1]}")

"""
 To Detect Anomalies in a new BGL System Log file
"""
# Load new BGL log file
def load_unlabelled_logs(filepath, max_lines=100000):
    logs = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i >= max_lines:
                break
            logs.append(line.strip())
    return logs

new_file_path = "/content/drive/MyDrive/New_BGL.log"
new_logs = load_unlabelled_logs(new_file_path)

# Generate embeddings for new logs
new_embeddings = get_bert_embeddings(new_logs, batch_size=500)
new_embeddings_tensor = torch.tensor(new_embeddings, dtype=torch.float32).to(device)

# Predict anomalies in the new file
new_predicted_labels = []
with torch.no_grad():
    for i in range(0, new_embeddings_tensor.shape[0], 500):
        batch = new_embeddings_tensor[i:i+500]
        predictions = classifier(batch)
        new_predicted_labels.append(torch.argmax(predictions, dim=1))
new_predicted_labels = torch.cat(new_predicted_labels).cpu().numpy()

# Save detected anomalies from the new file
new_anomaly_logs = [log for log, label in zip(new_logs, new_predicted_labels) if label == 1]
with open("/content/New_BGL_Anomalies.txt", "w", encoding='utf-8') as f:
    for log in new_anomaly_logs:
        f.write(log + "\n")

print("Anomalies detected in new BGL log file and saved to /content/New_BGL_Anomalies.txt")
