# ============================================================
# LSTM MODEL (WITH DROPOUT + PROPER GENERALIZATION CONTROL)
# ------------------------------------------------------------
# Purpose:
# This model implements an LSTM-based sentiment classifier
# with dropout regularization to prevent overfitting and
# ensure realistic performance on IMDb dataset.
#
# Key features:
# - Embedding layer for token representation
# - LSTM for sequence modeling
# - Dropout for regularization
# - Fully connected classifier
# ============================================================

import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=2):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        _, (hidden, _) = self.lstm(x)

        hidden = self.dropout(hidden[-1])

        return self.classifier(hidden)