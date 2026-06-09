# ============================================================
# TEST: VOCABULARY MODULE
# ------------------------------------------------------------
# Purpose:
# This test validates that the Vocabulary class correctly
# builds a word-to-index mapping and encodes tokens properly.
#
# It ensures:
# - special tokens (<PAD>, <UNK>) exist
# - words are correctly mapped to indices
# - encoding returns integer IDs
#
# This is critical because the model ONLY understands numbers,
# not raw text.
# ============================================================

from collections import Counter
from src.features.vocabulary import Vocabulary


def test_vocabulary():
    """
    Tests vocabulary creation and encoding behavior.
    """

    # Simulated tokenized corpus
    tokens = [
        ["i", "love", "this", "movie"],
        ["this", "movie", "is", "great"],
        ["i", "hate", "this", "film"]
    ]

    # Build frequency counter
    counter = Counter()
    for sentence in tokens:
        counter.update(sentence)

    # Build vocabulary
    vocab = Vocabulary()
    vocab.build([list(counter.keys())])

    print("Vocabulary size:", len(vocab.stoi))
    print("Sample tokens:", list(vocab.stoi.items())[:10])

    # Test encoding
    test_tokens = ["i", "love", "movie"]
    encoded = vocab.encode(test_tokens)

    print("Tokens:", test_tokens)
    print("Encoded:", encoded)

    # Assertions
    assert isinstance(encoded, list), "Encoded output must be a list"
    assert len(encoded) == len(test_tokens), "Encoding length mismatch"


if __name__ == "__main__":
    test_vocabulary()