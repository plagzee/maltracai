# train.py

import os
import re
import pandas as pd
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# ---------- Clean Text ---------- #
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    words = [w for w in text.split() if w not in stopwords.words("english")]
    return " ".join(words)

# ---------- Load SMS Dataset ---------- #
def load_sms_dataset():
    path = "data/sms/SMSSpamCollection"
    df = pd.read_csv(path, sep="\t", names=["label", "message"])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message'] = df['message'].apply(clean_text)
    return df

# ---------- Load SpamAssassin Dataset ---------- #
def load_spamassassin_dataset():
    base = "data/spamassassin"
    spam_path = os.path.join(base, "spam")
    ham_path = os.path.join(base, "easy_ham")

    def load_folder(folder_path, label):
        messages = []
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, 'r', encoding='latin-1') as f:
                        content = f.read()
                        messages.append((label, clean_text(content)))
                except:
                    continue
        return messages

    spam = load_folder(spam_path, 1)
    ham = load_folder(ham_path, 0)
    return pd.DataFrame(spam + ham, columns=["label", "message"])

# ---------- Vocabulary ---------- #
class TextVocab:
    def __init__(self, texts, max_size=8000):
        words = " ".join(texts).split()
        most_common = Counter(words).most_common(max_size - 2)
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        for i, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = i

    def encode(self, text, max_len=100):
        tokens = text.split()
        ids = [self.word2idx.get(w, 1) for w in tokens]
        return ids[:max_len] + [0] * (max_len - len(ids[:max_len]))

# ---------- Dataset Class ---------- #
class MessageDataset(Dataset):
    def __init__(self, messages, labels, vocab):
        self.messages = messages
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.vocab.encode(self.messages[idx])
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float)

# ---------- Model ---------- #
class ScamDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out).squeeze()

# ---------- Train Function ---------- #
def train_model(model, train_loader, val_loader, epochs=5):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                preds = model(x_batch)
                all_preds += (preds > 0.5).int().tolist()
                all_labels += y_batch.int().tolist()

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} / {epochs} - Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "scam_detector_model.pt")
    print("✅ Model saved to scam_detector_model.pt")

# ---------- Main ---------- #
if __name__ == "__main__":
    print("📥 Loading datasets...")
    sms_df = load_sms_dataset()
    spam_df = load_spamassassin_dataset()
    df = pd.concat([sms_df, spam_df]).sample(frac=1).reset_index(drop=True)

    print("📊 Building vocab...")
    vocab = TextVocab(df['message'].tolist())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['message'], df['label'], test_size=0.2, stratify=df['label']
    )

    train_ds = MessageDataset(train_texts.tolist(), train_labels.tolist(), vocab)
    val_ds = MessageDataset(val_texts.tolist(), val_labels.tolist(), vocab)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    print("🧠 Training model...")
    model = ScamDetector(vocab_size=len(vocab.word2idx))
    train_model(model, train_loader, val_loader)
