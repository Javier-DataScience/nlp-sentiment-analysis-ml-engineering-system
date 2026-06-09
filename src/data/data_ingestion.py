# ============================================================
# DATA INGESTION MODULE (UPDATED WITH IMDb SUPPORT)
# ------------------------------------------------------------
# Purpose:
# This module is responsible for loading raw data from a source.
# It now supports:
# - local CSV data
# - HuggingFace IMDb dataset
#
# Output:
# Returns raw dataset (text + label) ready for dataset.py
# ============================================================

from datasets import load_dataset


class DataIngestion:
    def __init__(self, config):
        """
        config example:
        {
            "data": {
                "source": "imdb"  # or "csv"
            }
        }
        """
        self.config = config
        self.source = config["data"]["source"]

    def load_data(self):
        """
        Loads dataset based on configuration.
        """

        # ========================================================
        # CASE 1: IMDb dataset (NEW)
        # ========================================================
        if self.source == "imdb":

            print("Loading IMDb dataset from HuggingFace...")

            train_data = load_dataset("imdb", split="train")
            test_data = load_dataset("imdb", split="test")

            print("IMDb loaded successfully")
            print("Train size:", len(train_data))
            print("Test size:", len(test_data))

            return train_data, test_data

        # ========================================================
        # CASE 2: CSV (your old logic - placeholder)
        # ========================================================
        elif self.source == "csv":

            print("Loading CSV dataset...")

            # keep your existing CSV logic here
            # (not modified in this step)

            raise NotImplementedError("CSV loader still unchanged")

        # ========================================================
        # UNKNOWN SOURCE
        # ========================================================
        else:
            raise ValueError(f"Unknown data source: {self.source}")