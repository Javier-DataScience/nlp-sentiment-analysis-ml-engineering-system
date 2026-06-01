"""
Vocabulary Module

This module builds and manages a vocabulary for NLP tasks.

It performs two main functions:
1. Build a mapping from tokens (words) to integer IDs.
2. Convert tokenized text into numerical format (encoding) and back (decoding).

Special tokens:
- <PAD>: used for padding sequences in batching
- <UNK>: used for unknown/unseen words
"""

class Vocabulary:
    def __init__(self):
        # token → id mapping
        self.stoi = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        # id → token mapping
        self.itos = {
            0: "<PAD>",
            1: "<UNK>"
        }

    def build(self, tokenized_texts):
        """
        Builds vocabulary from a list of tokenized sentences.

        Args:
            tokenized_texts (list[list[str]]): list of tokenized sentences
        """
        for tokens in tokenized_texts:
            for token in tokens:
                if token not in self.stoi:
                    idx = len(self.stoi)
                    self.stoi[token] = idx
                    self.itos[idx] = token

    def encode(self, tokens):
        """
        Converts a list of tokens into a list of integer IDs.

        Args:
            tokens (list[str])

        Returns:
            list[int]
        """
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

    def decode(self, ids):
        """
        Converts a list of IDs back into tokens.

        Args:
            ids (list[int])

        Returns:
            list[str]
        """
        return [self.itos.get(i, "<UNK>") for i in ids]

    def vocab_size(self):
        """
        Returns total vocabulary size.
        """
        return len(self.stoi)