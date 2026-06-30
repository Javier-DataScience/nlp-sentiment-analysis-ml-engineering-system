# ============================================================
# TEST: DATASET MODULE
# ------------------------------------------------------------
# PURPOSE:
# Validates the SentimentDataset class used in the production
# NLP pipeline.
#
# This version intentionally avoids external dependencies
# (HuggingFace, internet access, IMDb downloads) so that:
#
# - CI remains deterministic
# - Tests run offline
# - Execution is fast
# - No external services are required
#
# ARCHITECTURE:
#
# Mock text
#     ↓
# Tokenizer
#     ↓
# Vocabulary lookup
#     ↓
# Tensor conversion
#     ↓
# Model-ready sample
#
# MYPY NOTES:
# - Explicit Counter typing is used.
# - Method monkey-patching is intentionally performed to keep
#   tests completely offline and deterministic.
# ============================================================

from collections import Counter

import torch

from src.data.dataset import SentimentDataset
from src.features.tokenizer import SimpleTokenizer
from src.features.vocabulary import Vocabulary


def test_dataset():
    """
    Tests the complete dataset pipeline using
    deterministic local mock data.
    """

    tokenizer = SimpleTokenizer()

    # ========================================================
    # MOCK DATASET
    # ========================================================
    sample_data = [
        ("this movie was fantastic", 1),
        ("this movie was terrible", 0),
        ("i absolutely loved this film", 1),
        ("the acting was awful", 0),
    ]

    # ========================================================
    # BUILD TEST VOCABULARY
    # ========================================================
    counter: Counter[str] = Counter()

    for text, _ in sample_data:
        counter.update(tokenizer.tokenize(text))

    vocab = Vocabulary()

    vocab.build(list(counter.keys()))

    # ========================================================
    # MONKEY PATCH DATA LOADER
    # --------------------------------------------------------
    # Prevents the unit test from downloading IMDb from
    # HuggingFace during CI execution.
    # ========================================================
    original_loader = SentimentDataset._load_data

    SentimentDataset._load_data = (  # type: ignore[method-assign]
        lambda self, split: sample_data
    )

    try:

        # ====================================================
        # CREATE DATASET
        # ====================================================
        dataset = SentimentDataset(
            split="train",
            tokenizer=tokenizer,
            vocab=vocab,
        )

        sample = dataset[0]

        # ====================================================
        # ASSERTIONS
        # ====================================================
        assert "text" in sample
        assert "label" in sample

        assert isinstance(sample["text"], torch.Tensor)
        assert isinstance(sample["label"], torch.Tensor)

        assert sample["text"].dtype == torch.long
        assert sample["label"].dtype == torch.long

        assert len(dataset) == len(sample_data)

    finally:

        # ====================================================
        # RESTORE ORIGINAL IMPLEMENTATION
        # ====================================================
        SentimentDataset._load_data = original_loader  # type: ignore[method-assign]
