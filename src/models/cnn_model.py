# ============================================================
# TEXT CNN MODEL
# ------------------------------------------------------------
# PURPOSE:
# CNN-based sentiment classifier for NLP tasks.
#
# ARCHITECTURE:
# Input
# → Embedding Layer
# → Multiple 1D Convolutions
# → Global Max Pooling
# → Dropout
# → Fully Connected Layer
# → Output Classes
#
# WHY THIS MODEL:
# - Fast training
# - Strong NLP baseline
# - Captures local word patterns
# - Common benchmark in sentiment analysis
#
# REFERENCES:
# Yoon Kim (2014)
# "Convolutional Neural Networks for Sentence Classification"
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_classes,
        num_filters=100,
        filter_sizes=(3, 4, 5),
        dropout=0.5,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            num_filters * len(filter_sizes),
            num_classes,
        )

    def forward(self, x):

        # (batch_size, seq_len)
        embedded = self.embedding(x)

        # (batch_size, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)

        conv_outputs = []

        for conv in self.convs:

            feature_map = F.relu(conv(embedded))

            pooled = F.max_pool1d(
                feature_map,
                kernel_size=feature_map.shape[2],
            )

            pooled = pooled.squeeze(2)

            conv_outputs.append(pooled)

        concatenated = torch.cat(conv_outputs, dim=1)

        dropped = self.dropout(concatenated)

        return self.fc(dropped)
