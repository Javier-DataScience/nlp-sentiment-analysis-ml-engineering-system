"""
Tokenizer Module

This module is responsible for converting raw text into tokens.

It performs basic text preprocessing:
- Lowercasing
- Removing HTML tags
- Splitting text into words

Output is a list of clean tokens that can be used by the Vocabulary module.
"""

import re

class SimpleTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        """
        Converts raw text into a list of tokens.

        Args:
            text (str): raw input text

        Returns:
            list[str]: cleaned tokens
        """

        # lowercase
        text = text.lower()

        # remove html tags
        text = re.sub(r"<.*?>", " ", text)

        # keep only words (basic cleaning)
        text = re.sub(r"[^a-z\s]", " ", text)

        # split into tokens
        tokens = text.split()

        return tokens