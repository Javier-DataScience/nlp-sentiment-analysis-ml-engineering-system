"""
data_ingestion.py

PURPOSE:
- Create a clean, reproducible dataset pipeline
- Standardize dataset into:
    data/raw/train.csv
    data/raw/test.csv

RULES:
- NO notebooks dependency
- NO manual file management
- ALL folders are created automatically
"""

import os
import pandas as pd


class IMDBDataIngestion:

    def __init__(self, input_dir="data/raw"):
        self.input_dir = input_dir

    def _ensure_dirs(self):
        """
        Ensures required project data structure exists.
        """
        os.makedirs("data/raw", exist_ok=True)

    def load_dataset(self):
        """
        Loads dataset from project CSV files.
        """
        train_path = os.path.join(self.input_dir, "train.csv")
        test_path = os.path.join(self.input_dir, "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        return train_df, test_df

    def validate(self, train_df, test_df):
        """
        Basic sanity checks to ensure dataset integrity.
        """
        required_cols = ["text", "label"]

        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"Missing column in train: {col}")

            if col not in test_df.columns:
                raise ValueError(f"Missing column in test: {col}")

    def run(self):
        """
        Main entry point for ingestion pipeline.
        """
        print("Ensuring data structure exists...")
        self._ensure_dirs()

        print("Loading dataset from data/raw/...")

        train_df, test_df = self.load_dataset()

        print("Validating dataset...")
        self.validate(train_df, test_df)

        print("Train size:", len(train_df))
        print("Test size:", len(test_df))

        return train_df, test_df