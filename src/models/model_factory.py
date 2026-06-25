# ============================================================
# MODEL FACTORY (MULTI-MODEL NLP SYSTEM)
# ------------------------------------------------------------
# PURPOSE:
# Centralized model selection mechanism for the sentiment
# analysis project.
#
# SUPPORTED MODELS:
# - baseline
# - lstm
# - gru
# - cnn
# - bilstm
#
# This factory allows the training pipeline to instantiate
# models dynamically from configuration files.
#
# BENEFITS:
# - Single entry point for model creation
# - Easier experimentation
# - Cleaner training code
# - Extensible architecture for future models
# ============================================================

from src.models.baseline_model import BaselineModel
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.cnn_model import CNNModel
from src.models.bilstm_model import BiLSTMModel


def get_model(config):

    model_type = config["model"]["type"]

    # ========================================================
    # BASELINE MODEL
    # ========================================================
    if model_type == "baseline":

        return BaselineModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            num_classes=config["model"]["num_classes"],
        )

    # ========================================================
    # LSTM MODEL
    # ========================================================
    elif model_type == "lstm":

        return LSTMModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )

    # ========================================================
    # GRU MODEL
    # ========================================================
    elif model_type == "gru":

        return GRUModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )

    # ========================================================
    # CNN MODEL
    # ========================================================
    elif model_type == "cnn":

        return CNNModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            num_classes=config["model"]["num_classes"],
        )

    # ========================================================
    # BIDIRECTIONAL LSTM MODEL
    # ========================================================
    elif model_type == "bilstm":

        return BiLSTMModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================
    else:

        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported models: "
            f"['baseline', 'lstm', 'gru', 'cnn', 'bilstm']"
        )