# ============================================================
# BIDIRECTIONAL LSTM MODEL
# ------------------------------------------------------------
# PURPOSE:
# BiLSTM-based sentiment classifier for NLP tasks.
#
# ARCHITECTURE:
# Input
# → Embedding Layer
# → Bidirectional LSTM
# → Mean Pooling
# → Dropout
# → Fully Connected Layer
# → Output Classes
#
# WHY THIS MODEL:
# - Captures past and future context simultaneously
# - Usually stronger than vanilla LSTM
# - Widely used as a production NLP baseline
#
# OUTPUT DIMENSION:
# hidden_dim * 2 (forward + backward states)
# ============================================================

import torch.nn as nn


class BiLSTMModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        dropout=0.3,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_dim * 2,
            num_classes,
        )

    def forward(self, x):

        embedded = self.embedding(x)

        outputs, _ = self.lstm(embedded)

        pooled = outputs.mean(dim=1)

        pooled = self.dropout(pooled)

        return self.fc(pooled)
