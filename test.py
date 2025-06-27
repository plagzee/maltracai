# test.py

import torch
import torch.nn as nn
import pandas as pd
import re
from bs4 import BeautifulSoup
from collections import Counter
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import os

nltk.download('stopwords', quiet=True)

# --- Clean & Encode --- #
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    words = [w for w in text.split() if w not in stopwords.words("english")]
    return " ".join(words)

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

# --- Model --- #
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

# --- Load Training Data (for vocab only) --- #
def load_sms_dataset():
    path = "data/sms/SMSSpamCollection"
    df = pd.read_csv(path, sep="\t", names=["label", "message"])
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

# --- Prediction Function --- #
def predict_message(message, model, vocab):
    cleaned = clean_text(message)
    encoded = vocab.encode(cleaned)
    input_tensor = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        prob = model(input_tensor).item()
    return prob

# --- Chart --- #
def show_chart(probability):
    plt.figure(figsize=(5, 3))
    plt.bar(["Not Scam", "Scam"], [1 - probability, probability], color=["green", "red"])
    plt.ylim([0, 1])
    plt.ylabel("Probability")
    plt.title("🔍 Scam Prediction")
    plt.text(0, 1 - probability / 2, f"{(1 - probability)*100:.1f}%", ha='center', color='white', fontsize=12)
    plt.text(1, probability / 2, f"{probability*100:.1f}%", ha='center', color='white', fontsize=12)
    plt.tight_layout()
    plt.show()

# --- Main --- #
def main():
    print("🔁 Loading model and vocab...")

    sms_df = load_sms_dataset()
    spam_df = load_spamassassin_dataset()
    all_df = pd.concat([sms_df, spam_df]).sample(frac=1).reset_index(drop=True)

    vocab = TextVocab(all_df["message"].tolist())

    model = ScamDetector(vocab_size=len(vocab.word2idx))
    model.load_state_dict(torch.load("scam_detector_model.pt"))
    model.eval()

    print("✅ Ready to test messages! (type 'exit' to quit)")
    show = input("Do you want to see chart? (y/n): ").strip().lower() == 'y'

    while True:
        text = input("\n💬 Enter message:\n> ")
        if text.lower() == "exit":
            print("👋 Exiting...")
            break

        prob = predict_message(text, model, vocab)
        label = "SCAM ❌" if prob > 0.5 else "NOT SCAM ✅"
        print(f"🔍 Prediction: {label} ({prob * 100:.2f}%)")

        if show:
            show_chart(prob)

if __name__ == "__main__":
    main()
