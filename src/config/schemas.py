# ============================================================
# PYDANTIC CONFIGURATION SCHEMAS
# ------------------------------------------------------------
# PURPOSE:
# Provides strongly-typed configuration objects for the NLP
# sentiment analysis system.
#
# BENEFITS:
# - Runtime validation
# - IDE autocompletion
# - Safer refactoring
# - Better Airflow integration
# - Better cloud migration support
# - Future FastAPI compatibility
#
# DESIGN PRINCIPLES:
# - Non-invasive adoption
# - No behavior changes
# - Backward compatibility with existing dictionaries
# - Incremental migration
#
# ARCHITECTURE:
#
# AppConfig
# ├── ModelConfig
# └── TrainingConfig
# ============================================================

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """
    Model hyperparameters.
    """

    type: str
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_classes: int


class TrainingConfig(BaseModel):
    """
    Training hyperparameters.
    """

    lr: float
    batch_size: int
    epochs: int


class AppConfig(BaseModel):
    """
    Complete application configuration.
    """

    model: ModelConfig
    training: TrainingConfig
