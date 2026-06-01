"""
PREPROCESSING MODULE (NLP - PYTORCH PIPELINE)

This module handles all text preprocessing operations for the sentiment analysis system.

It ensures that raw text data is transformed into a clean, normalized format
that can be consumed by PyTorch-based models.

Key responsibilities:
- Text normalization (lowercasing, stripping noise)
- Basic cleaning (punctuation, extra spaces)
- Tokenization preparation
- Ensuring identical preprocessing for training and inference pipelines
"""

import re
import string


class TextPreprocessor:
    """
    Text preprocessing class for NLP sentiment analysis.
    """

    def __init__(self):
        """
        Initializes preprocessing utilities.
        """
        self.punctuation_table = str.maketrans("", "", string.punctuation)

    def clean_text(self, text: str) -> str:
        """
        Cleans raw input text.

        Args:
            text (str): Raw text input.

        Returns:
            str: Cleaned text.
        """

        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(self.punctuation_table)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def transform(self, text: str) -> str:
        """
        Full preprocessing pipeline for a single text input.

        Args:
            text (str): Raw text input.

        Returns:
            str: Processed text ready for vectorization or tokenization.
        """
        return self.clean_text(text)