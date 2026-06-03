"""
Dataset Module (Sentiment Analysis)

Pure dataset layer:
- Loads CSV
- Tokenizes text
- Encodes using vocabulary
- Returns tensors

NO dependencies on models or training logic.
"""

import pandas as pd
import torch


class SentimentDataset:
    def __init__(self, file_path, tokenizer, vocab):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        text = row["text"]
        label = row["label"]

        tokens = self.tokenizer.tokenize(text)
        encoded = self.vocab.encode(tokens)

        return {
            "text": torch.tensor(encoded, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }