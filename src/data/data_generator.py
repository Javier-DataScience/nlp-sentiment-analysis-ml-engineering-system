"""
data_generator.py

PURPOSE:
- Create initial dataset files for the ML pipeline
- Ensures data/raw/train.csv and test.csv exist
- Removes dependency on external/manual datasets
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DataGenerator:

    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir

    def _ensure_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_sample_dataset(self):
        """
        Temporary dataset generator (for development only)
        Replace later with real IMDB or production dataset.
        """
        data = {
            "text": [
                "I love this movie",
                "I hate this film",
                "Amazing story and acting",
                "Terrible movie, very boring",
                "Best film ever",
                "Worst experience",
                "I really enjoyed it",
                "Not good at all"
            ],
            "label": [1, 0, 1, 0, 1, 0, 1, 0]
        }

        df = pd.DataFrame(data)
        return df

    def split_and_save(self, df):
        """
        Split dataset and save into data/raw/
        """
        train_df, test_df = train_test_split(
            df,
            test_size=0.25,
            random_state=42,
            stratify=df["label"]
        )

        train_path = os.path.join(self.output_dir, "train.csv")
        test_path = os.path.join(self.output_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        return train_path, test_path

    def run(self):
        print("Generating dataset...")

        self._ensure_dirs()

        df = self.generate_sample_dataset()

        print("Splitting dataset...")

        train_path, test_path = self.split_and_save(df)

        print("Dataset created:")
        print("Train:", train_path)
        print("Test:", test_path)

        return train_path, test_path