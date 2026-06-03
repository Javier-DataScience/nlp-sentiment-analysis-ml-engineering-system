"""
config.py

============================================================
PURPOSE:
------------------------------------------------------------
Central configuration loader for all experiments.

This module ensures:
- reproducibility of experiments
- separation of code and configuration
- compatibility with MLflow tracking
- clean switching between models via YAML files

DESIGN PRINCIPLE:
------------------------------------------------------------
NO hyperparameters should be hardcoded in training code.
Everything must be defined in external YAML config files.
============================================================
"""

import yaml


def load_config(config_path: str):
    """
    Loads YAML configuration file.

    Args:
        config_path (str): path to YAML file

    Returns:
        dict: configuration dictionary
    """

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config