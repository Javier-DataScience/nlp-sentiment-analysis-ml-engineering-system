# ============================================================
# TEST: VOCABULARY MODULE
# ------------------------------------------------------------
# PURPOSE:
# Validates the production Vocabulary implementation.
#
# Ensures:
# - special tokens exist
# - token → id mapping works
# - unknown token handling works
# - reverse lookup works
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
# ============================================================

from collections import Counter

from src.features.vocabulary import Vocabulary


def test_vocabulary():
    """
    Tests vocabulary construction and lookup behavior.
    """

    tokens = [
        ["i", "love", "this", "movie"],
        ["this", "movie", "is", "great"],
        ["i", "hate", "this", "film"],
    ]

    counter = Counter()

    for sentence in tokens:
        counter.update(sentence)

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
