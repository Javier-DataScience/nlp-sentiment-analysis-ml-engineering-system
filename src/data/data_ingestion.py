# ============================================================
# DATA INGESTION MODULE (IMDB LOADER - STABLE VERSION)
# ------------------------------------------------------------
# PURPOSE:
# Loads IMDb dataset in a clean, reusable format for training.
#
# OUTPUT FORMAT:
# - texts: list[str]
# - labels: list[int]
#
# Labels:
# - 1 = positive
# - 0 = negative
# ============================================================

from datasets import load_dataset


def load_imdb(split="train"):
    """
    Loads IMDb dataset from Hugging Face.
    """

    dataset = load_dataset("imdb")

    data = dataset[split]

    texts = []
    labels = []

    for item in data:
        texts.append(item["text"])
        labels.append(item["label"])

    return texts, labels
