# ============================================================
# TEST: VOCABULARY MODULE
# ------------------------------------------------------------
# PURPOSE:
# Validates the production Vocabulary implementation used by
# the NLP sentiment analysis pipeline.
#
# THIS TEST ENSURES:
# - Special tokens exist.
# - Token → ID mapping works correctly.
# - Unknown token handling behaves as expected.
# - Reverse lookup (ID → token) works.
#
# IMPLEMENTATION DETAILS:
#
# The production implementation uses:
#
# token_to_id
# id_to_token
#
# instead of:
#
# stoi
# itos
#
# ARCHITECTURE:
#
# Raw tokens
#     ↓
# Counter
#     ↓
# Vocabulary.build()
#     ↓
# token_to_id mapping
#     ↓
# Reverse decoding
#
# DESIGN PRINCIPLES:
# - Fast execution (< 1 second)
# - No external dependencies
# - Deterministic behavior
# - CI/CD friendly
# ============================================================

from collections import Counter

from src.features.vocabulary import Vocabulary


def test_vocabulary():
    """
    Tests vocabulary construction and lookup behavior.
    """

    # ========================================================
    # MOCK TOKENIZED SENTENCES
    # ========================================================
    tokens = [
        ["i", "love", "this", "movie"],
        ["this", "movie", "is", "great"],
        ["i", "hate", "this", "film"],
    ]

    # ========================================================
    # BUILD TOKEN COUNTS
    # ========================================================
    counter: Counter[str] = Counter()

    for sentence in tokens:
        counter.update(sentence)

    # ========================================================
    # BUILD VOCABULARY
    # ========================================================
    vocab = Vocabulary()

    vocab.build(list(counter.keys()))

    # ========================================================
    # SPECIAL TOKENS
    # ========================================================
    assert "<PAD>" in vocab.token_to_id
    assert "<UNK>" in vocab.token_to_id

    # ========================================================
    # KNOWN TOKEN LOOKUP
    # ========================================================
    movie_id = vocab.get("movie")

    assert isinstance(movie_id, int)

    # ========================================================
    # UNKNOWN TOKEN LOOKUP
    # ========================================================
    unknown_id = vocab.get("some_random_word")

    assert unknown_id == vocab.token_to_id["<UNK>"]

    # ========================================================
    # REVERSE LOOKUP
    # ========================================================
    decoded = vocab.decode(movie_id)

    assert decoded == "movie"
