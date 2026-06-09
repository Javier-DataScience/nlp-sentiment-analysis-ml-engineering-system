# ============================================================
# TEST: MODEL FACTORY
# ------------------------------------------------------------
# Purpose:
# This test validates the model_factory module, which is
# responsible for selecting and instantiating ML models
# based on configuration.
#
# It ensures:
# - correct model is created from config
# - model architecture is valid
# - forward pass executes without errors
# - output shape matches classification task
#
# This is a key step before training and MLflow integration.
# ============================================================

import torch
from src.models.model_factory import get_model


def test_model_factory():
    """
    Tests model creation and forward pass.
    """

    # Configuration dictionary (simulating config.yaml)
    config = {
        "model": {
            "type": "baseline",
            "vocab_size": 5000,
            "embed_dim": 128,
            "num_classes": 2
        }
    }

    # Create model via factory
    model = get_model(config)

    print("Model created:")
    print(model)

    # Dummy input: batch_size=2, seq_len=4
    x = torch.randint(0, 5000, (2, 4))

    # Forward pass
    output = model(x)

    print("Output shape:", output.shape)

    # Assertions
    assert output.shape == (2, 2), "Unexpected output shape"


if __name__ == "__main__":
    test_model_factory()