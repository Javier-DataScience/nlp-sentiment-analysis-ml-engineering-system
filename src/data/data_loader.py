"""
DATA LOADER MODULE

This module is responsible for loading raw datasets in a centralized and consistent way.

It ensures that all components of the ML system (EDA notebooks, training pipeline, and inference pipeline)
use the same data access logic, preventing duplication and inconsistencies.

Responsibilities:
- Load raw CSV or structured datasets
- Provide reusable functions for train/test loading
- Serve as the single entry point for data access across the project
"""

import os
import pandas as pd


class DataLoader:
    """
    Centralized data loading class for the ML system.
    """

    def __init__(self, data_path: str):
        """
        Initializes the DataLoader with the base path of the dataset.

        Args:
            data_path (str): Path to the folder containing data files.
        """
        self.data_path = data_path

    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Loads a CSV file from the data directory.

        Args:
            filename (str): Name of the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        file_path = os.path.join(self.data_path, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return pd.read_csv(file_path)

    def load_train_test(self, train_file: str, test_file: str):
        """
        Loads train and test datasets.

        Args:
            train_file (str): Training dataset filename.
            test_file (str): Test dataset filename.

        Returns:
            tuple: (train_df, test_df)
        """
        train_df = self.load_csv(train_file)
        test_df = self.load_csv(test_file)

        return train_df, test_df