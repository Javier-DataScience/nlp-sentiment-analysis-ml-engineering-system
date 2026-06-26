# ============================================================
# VOCABULARY BUILDER (PRODUCTION + TRAINING COMPATIBLE)
# ------------------------------------------------------------
# PURPOSE:
# Builds a clean vocabulary from the IMDb dataset using HuggingFace.
#
# KEY FEATURES:
# - Stable deterministic vocabulary creation
# - Frequency filtering (min_freq)
# - Compatible with SentimentDataset + training pipeline
# - Safe importable function for training scripts
#
# OUTPUT:
# - Vocabulary object ready for model embedding layers
# ============================================================

from collections import Counter
from datasets import load_dataset

from src.features.tokenizer import SimpleTokenizer
from src.features.vocabulary import Vocabulary


def build_vocabulary_from_csv(csv_path=None, split="train", min_freq=2):
    """
    Vocabulary builder (compatible interface for training pipeline).

    NOTE:
    - csv_path kept for backward compatibility (ignored if None)
    - primary source is HuggingFace IMDb dataset

    Args:
        csv_path (str): optional legacy argument (ignored)
        split (str): "train" or "test"
        min_freq (int): minimum token frequency threshold

    Returns:
        Vocabulary: fitted vocabulary object
    """

    print(f"Building vocabulary from IMDb ({split})...")

    dataset = load_dataset("imdb")[split]

    tokenizer = SimpleTokenizer()
    counter = Counter()

    # ========================================================
    # STEP 1: TOKENIZE ENTIRE CORPUS
    # ========================================================
    for item in dataset:
        tokens = tokenizer.tokenize(item["text"])
        counter.update(tokens)

    # ========================================================
    # STEP 2: FILTER LOW-FREQUENCY TOKENS
    # ========================================================
    filtered_tokens = [token for token, freq in counter.items() if freq >= min_freq]

    # ========================================================
    # STEP 3: BUILD VOCABULARY
    # ========================================================
    vocab = Vocabulary()
    vocab.build(filtered_tokens)

    print("Vocabulary built successfully")
    print("Vocab size:", len(vocab.token_to_id))

    return vocab


# ============================================================
# LOCAL DEBUG TEST
# ============================================================
if __name__ == "__main__":

    vocab = build_vocabulary_from_csv(split="train")

    sample = list(vocab.token_to_id.items())[:20]

    print("Sample vocab:", sample)
