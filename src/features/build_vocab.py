"""
build_vocab.py

This script builds a full vocabulary from the IMDB training dataset.
It replaces toy vocab construction and ensures that most words are NOT mapped to <UNK>.

It uses:
- SimpleTokenizer to tokenize text
- Vocabulary class to store word-to-index mapping

Output:
- vocab object that can be saved and reused in dataset pipeline
"""

import pandas as pd
from collections import Counter

from src.features.tokenizer import SimpleTokenizer
from src.features.vocabulary import Vocabulary


def build_vocabulary_from_csv(csv_path, min_freq=2):
    """
    Builds a vocabulary from IMDB training data.

    Args:
        csv_path (str): path to train.csv
        min_freq (int): minimum frequency for a word to be included

    Returns:
        Vocabulary: fitted vocabulary object
    """

    df = pd.read_csv(csv_path)

    tokenizer = SimpleTokenizer()

    counter = Counter()

    # Step 1: tokenize all texts and count words
    for text in df["text"]:
        tokens = tokenizer.tokenize(text)
        counter.update(tokens)

    # Step 2: filter by frequency
    filtered_tokens = [
        token for token, freq in counter.items()
        if freq >= min_freq
    ]

    # Step 3: build vocabulary
    vocab = Vocabulary()
    vocab.build([tokenizer.tokenize(text) for text in df["text"]])  # single corpus list

    return vocab


if __name__ == "__main__":
    vocab = build_vocabulary_from_csv("data/raw/train.csv")

    print("Vocab size:", len(vocab.stoi))
    print("Sample vocab:", list(vocab.stoi.items())[:20])