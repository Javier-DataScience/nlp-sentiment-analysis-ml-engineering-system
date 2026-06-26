# ============================================================
# TEST: DATASET MODULE
# ------------------------------------------------------------
# PURPOSE:
# Validates the SentimentDataset class used in the production
# NLP pipeline.
#
# This test ensures:
# - HuggingFace IMDb loading works
# - tokenization works correctly
# - vocabulary encoding works correctly
# - output format matches training expectations
# - tensors are returned for both text and labels
#
# ARCHITECTURE:
# Raw text
#     ↓
# Tokenizer
#     ↓
# Vocabulary lookup
#     ↓
# Tensor conversion
#     ↓
# Model-ready sample
# ============================================================

from collections import Counter

import torch

from src.data.dataset import SentimentDataset
from src.features.tokenizer import SimpleTokenizer
from src.features.vocabulary import Vocabulary
from src.data.data_ingestion import load_imdb


def test_dataset():
    """
    Tests the complete dataset pipeline.
    """

    tokenizer = SimpleTokenizer()

    # ========================================================
    # BUILD A SMALL TEST VOCABULARY
    # ========================================================
    texts, _ = load_imdb("train")

    counter = Counter()

    # Only use a small subset to keep tests fast
    for text in texts[:100]:
        counter.update(tokenizer.tokenize(text))

    vocab = Vocabulary()
    vocab.build(list(counter.keys()))

    # ========================================================
    # CREATE DATASET
    # ========================================================
    dataset = SentimentDataset(split="train", tokenizer=tokenizer, vocab=vocab)

    sample = dataset[0]

    # ========================================================
    # ASSERTIONS
    # ========================================================
    assert "text" in sample
    assert "label" in sample

    assert isinstance(sample["text"], torch.Tensor)
    assert isinstance(sample["label"], torch.Tensor)

    assert sample["text"].dtype == torch.long
    assert sample["label"].dtype == torch.long

    assert len(dataset) > 0
