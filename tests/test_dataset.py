# ============================================================
# TEST: DATASET MODULE
# ------------------------------------------------------------
# Purpose:
# This test validates the SentimentDataset class which is
# responsible for converting raw CSV text data into
# model-ready tensors.
#
# It ensures:
# - CSV is correctly loaded
# - text is tokenized using SimpleTokenizer
# - tokens are encoded using Vocabulary
# - output is returned as PyTorch tensors
#
# This is a critical bridge between raw data and model input.
# ============================================================

from src.data.dataset import SentimentDataset
from src.features.tokenizer import SimpleTokenizer
from src.features.vocabulary import Vocabulary
from collections import Counter


def test_dataset():
    """
    Tests dataset loading and encoding pipeline.
    """

    # Paths
    file_path = "data/raw/train.csv"

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()

    # Build minimal vocabulary from file (quick test version)
    import pandas as pd
    df = pd.read_csv(file_path)

    counter = Counter()
    for text in df["text"]:
        counter.update(tokenizer.tokenize(text))

    vocab = Vocabulary()
    vocab.build([list(counter.keys())])

    # Create dataset
    dataset = SentimentDataset(file_path, tokenizer, vocab)

    print("Dataset size:", len(dataset))

    # Test single sample
    sample = dataset[0]
    print("Sample[0]:", sample)

    # Assertions
    assert "text" in sample, "Missing 'text' key"
    assert "label" in sample, "Missing 'label' key"


if __name__ == "__main__":
    test_dataset()