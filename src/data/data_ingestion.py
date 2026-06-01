# data_ingestion.py
# This module is responsible for downloading and preparing the IMDb dataset
# using HuggingFace Datasets (production-safe, cloud-ready, no torchtext dependency).

import os
import pandas as pd
from datasets import load_dataset


class IMDBDataIngestion:
    """
    Downloads IMDb dataset from HuggingFace and saves train/test splits as CSV.
    """

    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        dataset = load_dataset("imdb")
        return dataset

    def save_as_csv(self):
        dataset = self.load_data()

        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])

        train_path = os.path.join(self.output_dir, "train.csv")
        test_path = os.path.join(self.output_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        return train_path, test_path