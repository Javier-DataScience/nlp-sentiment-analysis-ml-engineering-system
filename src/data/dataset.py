"""
Dataset Module (Sentiment Analysis)

This module defines a PyTorch Dataset for sentiment classification.

Responsibilities:
- Load CSV data (text + label)
- Tokenize text using an external tokenizer
- Convert tokens to IDs using an external vocabulary
- Return model-ready samples (no padding, no batching)

IMPORTANT:
- This dataset does NOT build vocabulary
- This dataset does NOT perform batching
- This dataset only returns single encoded samples
"""

import pandas as pd
import torch

class SentimentDataset:
    def __init__(self, file_path, tokenizer, vocab):
        """
        Args:
            file_path (str): Path to CSV file with columns [text, label]
            tokenizer (object): Tokenizer with .tokenize() method
            vocab (Vocabulary): Vocabulary with .encode() method
        """
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = row["text"]
        label = row["label"]

        # Step 1: tokenize
        tokens = self.tokenizer.tokenize(text)

        # Step 2: convert tokens → ids
        encoded = self.vocab.encode(tokens)

        return {
            "text": torch.tensor(encoded, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }