# predict.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords
import nltk
import pandas as pd

nltk.download('stopwords')

# ----------- Clean Text ----------- #
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    words = [w for w in text.split() if w not in stopwords.words("english")]
    return " ".join(words)

# ----------- Vocabulary ----------- #
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

# ----------- Model ----------- #
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

# ----------- Load Training Data for Vocab ----------- #
def load_sms_dataset():
    path = "data/sms/SMSSpamCollection"
    df = pd.read_csv(path, sep="\t", names=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["message"] = df["message"].apply(clean_text)
    return df

def load_spamassassin_dataset():
    spam_dir = "data/spamassassin/spam"
    ham_dir = "data/spamassassin/easy_ham"

    def load_folder(folder, label):
        data = []
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="latin-1") as f:
                        text = f.read()
                        data.append((label, clean_text(text)))
                except:
                    continue
        return pd.DataFrame(data, columns=["label", "message"])

    spam_df = load_folder(spam_dir, 1)
    ham_df = load_folder(ham_dir, 0)
    return pd.concat([spam_df, ham_df]).reset_index(drop=True)

# ----------- Predict ----------- #
def predict(message, model, vocab):
    model.eval()
    cleaned = clean_text(message)
    encoded = vocab.encode(cleaned)
    input_tensor = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        prob = model(input_tensor).item()
    return prob

# ----------- Matplotlib Chart ----------- #
def show_prediction(message, probability):
    plt.figure(figsize=(6, 4))
    plt.bar(["Not Scam", "Scam"], [1 - probability, probability], color=["green", "red"])
    plt.title("📊 MaltracAI Prediction")
    plt.ylim([0, 1])
    plt.ylabel("Probability")
    plt.text(0, 1 - probability / 2, f"{(1 - probability) * 100:.1f}%", ha='center', va='center', color='white', fontsize=12)
    plt.text(1, probability / 2, f"{probability * 100:.1f}%", ha='center', va='center', color='white', fontsize=12)
    plt.tight_layout()
    plt.show()

# ----------- Main ----------- #
if __name__ == "__main__":
    import os

    print("📥 Loading datasets for vocab building...")
    sms_df = load_sms_dataset()
    spam_df = load_spamassassin_dataset()
    all_df = pd.concat([sms_df, spam_df]).sample(frac=1).reset_index(drop=True)

    print("📖 Building vocabulary...")
    vocab = TextVocab(all_df["message"].tolist())

    print("🧠 Loading trained model...")
    model = ScamDetector(vocab_size=len(vocab.word2idx))
    model.load_state_dict(torch.load("scam_detector_model.pt"))
    model.eval()

    # 🎯 Get user input
    user_input = input("\n💬 Enter a message to analyze:\n> ")
    probability = predict(user_input, model, vocab)

    label = "SCAM ❌" if probability > 0.5 else "NOT SCAM ✅"
    print(f"\n🔍 Prediction: {label} ({probability*100:.2f}%)")

    show_prediction(user_input, probability)
