"""
Baseline Text Classification Model

Simple sentiment classifier using:
- Embedding layer
- Mean pooling
- Linear classifier
"""

import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        output = self.classifier(pooled)
        return output
