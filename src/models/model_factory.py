"""
model_factory.py

============================================================
PURPOSE:
------------------------------------------------------------
This module acts as a centralized model selector for the
entire NLP sentiment analysis system.

It is responsible for:
- Selecting the correct model architecture based on config
- Instantiating the model with the correct parameters
- Keeping training code fully decoupled from model definitions

IMPORTANT DESIGN PRINCIPLE:
------------------------------------------------------------
- Models are NOT defined here.
- Models are ONLY imported and instantiated.
- Adding a new model should NOT require changing training logic.
- Only this factory maps "config → model class".

EXPECTED USAGE:
------------------------------------------------------------
In train.py:

    from src.models.model_factory import get_model
    model = get_model(config)

This ensures:
- reproducibility
- scalability
- clean MLflow tracking
- safe multi-model experimentation

============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

from src.models.baseline_model import BaselineModel


# ============================================================
# FACTORY FUNCTION
# ============================================================

def get_model(config):
    """
    Creates and returns a model instance based on configuration.

    Args:
        config (dict): Loaded YAML configuration containing:
            - model.type (str): model name (e.g. "baseline")
            - model.vocab_size (int)
            - model.embed_dim (int)
            - model.num_classes (int)

    Returns:
        torch.nn.Module: instantiated model

    Raises:
        ValueError: if model type is not supported
    """

    model_type = config["model"]["type"]

    # ========================================================
    # BASELINE MODEL
    # ========================================================
    if model_type == "baseline":

        return BaselineModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            num_classes=config["model"]["num_classes"]
        )

    # ========================================================
    # UNKNOWN MODEL TYPE
    # ========================================================
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available models: ['baseline']"
        )