# ============================================================
# MODEL FACTORY (UPDATED FOR MULTI-MODEL SYSTEM)
# ------------------------------------------------------------
# Purpose:
# Centralized model selection system that allows switching
# between multiple architectures using config.
#
# Supported models:
# - baseline
# - lstm
# - gru
#
# This factory creates the appropriate model based on the
# configuration provided by the training pipeline.
# ============================================================

from src.models.baseline_model import BaselineModel
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel


def get_model(config):

    model_type = config["model"]["type"]

    # ---------------- BASELINE MODEL ----------------
    if model_type == "baseline":

        return BaselineModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            num_classes=config["model"]["num_classes"]
        )

    # ---------------- LSTM MODEL ----------------
    elif model_type == "lstm":

        return LSTMModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"]
        )

    # ---------------- GRU MODEL ----------------
    elif model_type == "gru":

        return GRUModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"]
        )

    # ---------------- ERROR HANDLING ----------------
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported: ['baseline', 'lstm', 'gru']"
        )