# ============================================================
# GRU MODEL (WITH DROPOUT + STABLE GENERALIZATION)
# ------------------------------------------------------------
# Purpose:
# GRU-based sentiment classifier optimized for sequence
# modeling with dropout regularization to reduce overfitting.
#
# Key features:
# - Embedding layer
# - GRU sequence encoder
# - Dropout regularization
# - Linear classification head
# ============================================================

import torch.nn as nn


class GRUModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=2):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.gru = nn.GRU(
            input_size=embed_dim, hidden_size=hidden_dim, batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        _, hidden = self.gru(x)

        hidden = self.dropout(hidden[-1])

        return self.classifier(hidden)
