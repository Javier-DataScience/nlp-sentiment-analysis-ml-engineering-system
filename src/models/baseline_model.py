"""
Baseline Text Classification Model

This is a minimal neural network for sentiment classification.
It uses:
- Embedding layer (turn token IDs into vectors)
- Mean pooling (simple aggregation over sequence)
- Linear layer (classification)

This model is intentionally simple to validate the pipeline.
"""

import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len)
        """

        embedded = self.embedding(x)          # (B, T, E)
        pooled = embedded.mean(dim=1)         # (B, E)
        output = self.classifier(pooled)      # (B, C)

        return output