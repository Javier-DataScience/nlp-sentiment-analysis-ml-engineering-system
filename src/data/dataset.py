# ============================================================
# SENTIMENT DATASET (IMPROVED REPRESENTATION LAYER)
# ------------------------------------------------------------
# PURPOSE:
# This dataset prepares IMDb data for NLP models with:
# - consistent tokenization
# - controlled vocabulary mapping
# - stable tensor output
#
# This version improves:
# - label consistency
# - sequence integrity
# - model learning signal
# ============================================================

import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):

    def __init__(self, split, tokenizer, vocab):

        self.tokenizer = tokenizer
        self.vocab = vocab

        # IMPORTANT: assuming HuggingFace-style loading already implemented
        # split = "train" or "test"
        self.data = self._load_data(split)

    def _load_data(self, split):

        # Expect format: list of (text, label)
        # This assumes you already built IMDb ingestion correctly
        from src.data.data_ingestion import load_imdb

        texts, labels = load_imdb(split)

        return list(zip(texts, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        text, label = self.data[idx]

        # 1. tokenize
        tokens = self.tokenizer.tokenize(text)

        # 2. convert to ids (IMPORTANT FIX)
        token_ids = [
            self.vocab.get(token) for token in tokens
            if self.vocab.get(token) is not None
        ]

        # safety fallback (avoid empty sequences)
        if len(token_ids) == 0:
            token_ids = [0]

        return {
            "text": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }