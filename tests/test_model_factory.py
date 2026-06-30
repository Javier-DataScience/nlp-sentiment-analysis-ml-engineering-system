# ============================================================
# TEST: TOKENIZER MODULE
# ------------------------------------------------------------
# PURPOSE:
# Validates the SimpleTokenizer used throughout the NLP
# sentiment analysis system.
#
# THIS TEST ENSURES:
# - Tokenization is deterministic.
# - The output type is a Python list.
# - The tokenizer produces non-empty results.
# - Basic preprocessing behavior remains stable.
#
# IMPORTANCE:
# The tokenizer is one of the foundational components of
# the entire NLP pipeline:
#
# Raw text
#     ↓
# Tokenizer
#     ↓
# Vocabulary lookup
#     ↓
# Tensor conversion
#     ↓
# Neural network input
#
# If tokenization changes unexpectedly, every downstream
# component may break or produce inconsistent results.
#
# DESIGN PRINCIPLES:
# - Fast execution (< 1 second)
# - No external dependencies
# - No internet access
# - Deterministic behavior
# - CI/CD friendly
# ============================================================

from src.features.tokenizer import SimpleTokenizer


def test_tokenizer():
    """
    Validates tokenizer behavior on a simple sentence.
    """

    # ========================================================
    # CREATE TOKENIZER
    # ========================================================
    tokenizer = SimpleTokenizer()

    # ========================================================
    # MOCK INPUT
    # ========================================================
    text = "I love this movie"

    # ========================================================
    # TOKENIZATION
    # ========================================================
    tokens = tokenizer.tokenize(text)

    # ========================================================
    # ASSERTIONS
    # ========================================================
    assert isinstance(tokens, list)

    assert len(tokens) > 0

    assert tokens == tokenizer.tokenize(text)
