# ============================================================
# MODEL REGISTRY (CENTRAL MODEL ZOO)
# ------------------------------------------------------------
# PURPOSE:
# Central registry containing all supported NLP models.
#
# RESPONSIBILITIES:
# - Provide a single source of truth for available models
# - Instantiate models from their names
# - Keep the system extensible for future architectures
#
# SUPPORTED MODELS:
# - baseline
# - lstm
# - gru
# - cnn
# - bilstm
#
# INPUTS:
# - model name
# - configuration dictionary
#
# OUTPUTS:
# - instantiated PyTorch model
#
# DESIGN PRINCIPLES:
# - Explicit model registration
# - No hidden architectures
# - Cloud-ready extensibility
# - Consistency with model_factory.py
#
# FUTURE EXTENSIONS:
# - transformer
# - bert
# - distilbert
# - roberta
# - llama-based classifiers
# ============================================================

from src.models.baseline_model import BaselineModel
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.cnn_model import CNNModel
from src.models.bilstm_model import BiLSTMModel

MODEL_REGISTRY = [
    "baseline",
    "lstm",
    "gru",
    "cnn",
    "bilstm",
]


def get_model_by_name(name, config):
    """
    Build a model from its registered name.

    Parameters
    ----------
    name : str
        Model identifier.

    config : dict
        Configuration dictionary.

    Returns
    -------
    torch.nn.Module
        Instantiated model.
    """

    if name == "baseline":

        return BaselineModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            num_classes=config["model"]["num_classes"],
        )

    elif name == "lstm":

        return LSTMModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )

    elif name == "gru":

        return GRUModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )

    elif name == "cnn":

        return CNNModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            num_classes=config["model"]["num_classes"],
        )

    elif name == "bilstm":

        return BiLSTMModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )

    else:

        raise ValueError(
            f"Unknown model: {name}. " f"Supported models: {MODEL_REGISTRY}"
        )
