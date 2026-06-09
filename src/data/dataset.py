# ============================================================
# SENTIMENT DATASET (IMDb - HuggingFace FIXED VERSION)
# ------------------------------------------------------------
# PURPOSE:
# Fully stable dataset loader using HuggingFace IMDb.
# Ensures correct train/test sizes and consistent encoding.
# ============================================================

from datasets import load_dataset
import torch


class SentimentDataset:

    def __init__(self, split, tokenizer, vocab, max_len=256):

        # Load IMDb directly from HuggingFace
        dataset = load_dataset("imdb", split=split)

        self.texts = dataset["text"]
        self.labels = dataset["label"]

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def encode(self, text):

        tokens = self.tokenizer.tokenize(text)
        ids = self.vocab.encode(tokens)

        # truncate only (no padding here)
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]

        return ids

    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]

        ids = self.encode(text)

        return {
            "text": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }