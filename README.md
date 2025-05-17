# AnomLogBERT: Deep Learning-Based Anomaly Detection in System Logs

## 🧠 Project Overview

**AnomLogBERT** is a deep learning framework for **automated anomaly detection in system log files**, applicable to both **Blue Gene/L (BGL)** and **Linux** environments. It leverages the power of transformer-based language models (`all-MiniLM-L12-v2`) for semantic understanding of logs and a feedforward neural network for classification.

This project is designed to enhance system reliability by detecting security threats, hardware failures, and abnormal behaviors in logs without relying on traditional rule-based or parsing approaches.

---

## 🧾 Key Features

- ✅ Log-type agnostic (supports BGL and Linux logs)
- 🧠 Uses pre-trained Sentence-BERT for semantic embedding
- 🔍 Custom regular-expression-based log preprocessing
- 📊 Class-weighted loss handling for imbalanced datasets
- 🧪 Evaluation with Accuracy, Precision, Recall, and F1-Score
- 🚨 Real-time anomaly detection in new unlabeled log files
- 💾 Saves detected anomalies for further analysis

---

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
pip install torch transformers scikit-learn matplotlib
```

### 2. Setup Google Drive (if using Colab)

Mount Google Drive to access log files:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 3. File Paths

Ensure the paths to the following files are correct in your script:

- `/content/drive/MyDrive/BGL.log`
- `/content/drive/MyDrive/Linux_labeled_full.txt`
- `/content/drive/MyDrive/New_BGL.log`

---

## 🚀 How It Works

1. **Preprocessing:** 
   - Replaces dynamic tokens (timestamps, versions, node IDs) with static placeholders.
2. **Embedding Generation:** 
   - Converts logs to semantic vectors using Sentence-BERT (`all-MiniLM-L12-v2`).
3. **Classification:**
   - Classifies embeddings into normal or anomaly using a custom feedforward neural network.
4. **Evaluation:**
   - Computes accuracy, precision, recall, F1.
5. **Anomaly Extraction:**
   - Saves predicted anomalies to `.txt` files for investigation.

---

## 📊 Sample Performance Metrics

### 📌 BGL Logs:
- **Accuracy:** 98.95%
- **Precision:** 99.91%
- **Recall:** 95.72%
- **F1-Score:** 97.77%

### 📌 Linux Logs:
- **Accuracy:** 99.95%
- **Precision:** 100.0%
- **Recall:** 99.69%
- **F1-Score:** 99.84%

---

## 🛠 Model Architecture

### ➤ BERT Encoder
- `all-MiniLM-L12-v2` Sentence-BERT
- Output: 384-dimensional embeddings

### ➤ Neural Network
- Input: 384-dim BERT embedding
- Hidden Layers: 128 → 64 (BGL) / 256 → 128 (Linux)
- Output: 2-class softmax

---

## 📈 Confusion Matrix Example (Linux)

```
               Predicted: Normal    Predicted: Anomaly
Actual: Normal       6338              0
Actual: Anomaly      4                 1329
```

---

## 🧪 Datasets Used

- **BGL (Blue Gene/L) Logs:** Supercomputer log with hardware failure traces.
- **Linux Logs:** Labeled system logs with normal and anomaly entries.

---

## ✍️ Authors

- Siddharth Yadav
- Ayush Shukla
- Sujal Chhajed

---

## 📚 References

This project is inspired by research works including:

- LAnoBERT: [Applied Soft Computing, 2023](https://doi.org/10.1016/j.asoc.2023.110689)
- DSGN: [Information Sciences, 2024](https://doi.org/10.1016/j.ins.2024.121174)
- AutoLog, LogFormer, LogGT, and others (full reference list in report)

---

## 📚 About the BGL Dataset

**BGL (Blue Gene/L)** is an open dataset of logs collected from a BlueGene/L supercomputer system at Lawrence Livermore National Labs (LLNL), featuring 131,072 processors and 32,768GB of memory.

- The logs include **alert and non-alert messages**.
- The **first column** of the log uses `"-"` to indicate non-alert (normal) messages, and other strings indicate alert (anomalous) messages.
- The dataset is ideal for research in **alert detection, anomaly prediction, and log analysis**.

🔗 **Project Page:** [USENIX CFDR](https://www.usenix.org/cfdr-data#hpc4)  
📥 **Download Logs:** [Loghub GitHub Repo](https://github.com/logpai/loghub)

### 📖 Citation

If you use this dataset, please cite:

- Adam J. Oliner, Jon Stearley.  
  *What Supercomputers Say: A Study of Five System Logs*, IEEE DSN 2007.  
  [DOI: 10.1109/DSN.2007.56](http://ieeexplore.ieee.org/document/4273008/)

- Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu.  
  *Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics*, ISSRE 2023.  
  [arXiv:2008.06448](https://arxiv.org/abs/2008.06448)
