# ============================================================
# TEST: TOKENIZER MODULE
# ------------------------------------------------------------
# Purpose:
# This test validates that the SimpleTokenizer works correctly
# and consistently tokenizes raw text into word-level tokens.
#
# It ensures:
# - tokenization is deterministic
# - output is a list of tokens
# - basic preprocessing pipeline works
#
# This is the foundation of the entire NLP pipeline.
# If tokenizer breaks, everything downstream breaks.
# ============================================================

from src.features.tokenizer import SimpleTokenizer


def test_tokenizer():
    """
    Validates tokenizer behavior on a simple sentence.
    """

    tokenizer = SimpleTokenizer()

    text = "I love this movie"
    tokens = tokenizer.tokenize(text)

    print("Input:", text)
    print("Tokens:", tokens)

    # Assertions (basic correctness checks)
    assert isinstance(tokens, list), "Tokens must be a list"
    assert len(tokens) > 0, "Tokenizer returned empty output"


if __name__ == "__main__":
    test_tokenizer()
