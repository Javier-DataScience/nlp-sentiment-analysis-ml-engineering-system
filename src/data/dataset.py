"""
SentimentDataset Module

This module defines a PyTorch Dataset class for sentiment analysis.
It loads text data from a CSV file and converts it into a format
compatible with PyTorch DataLoader for training and inference.

Responsibilities:
- Load dataset from CSV file
- Preprocess text (basic cleaning if needed)
- Convert labels into tensors
- Provide __len__ and __getitem__ methods for PyTorch
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis.
    Each sample consists of:
        - text (string)
        - label (0 or 1)
    """

    def __init__(self, file_path: str):
        """
        Initialize dataset by loading CSV file.

        Args:
            file_path (str): Path to CSV file containing:
                             columns = [text, label]
        """
        self.data = pd.read_csv(file_path)

        # Basic validation
        assert "text" in self.data.columns
        assert "label" in self.data.columns

    def __len__(self):
        """
        Returns:
            int: Number of samples in dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns one sample at index idx.

        Returns:
            dict: {
                "text": str,
                "label": torch.Tensor
            }
        """
        row = self.data.iloc[idx]

        text = row["text"]
        label = int(row["label"])

        return {
            "text": text,
            "label": torch.tensor(label, dtype=torch.long)
        }