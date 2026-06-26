# ============================================================
# INFERENCE MODULE (CHAMPION-BASED SENTIMENT PREDICTION)
# ------------------------------------------------------------
# PURPOSE:
# Central inference engine for the NLP sentiment analysis
# system.
#
# RESPONSIBILITIES:
# - Load the locked champion model
# - Load serialized vocabulary
# - Load model metadata
# - Reconstruct the architecture deterministically
# - Reuse the training tokenizer
# - Produce sentiment predictions
#
# INPUTS:
# - artifacts/champion.json
# - artifacts/vocab.pkl
# - artifacts/<model>_metadata.pkl
# - models/<model>.pt
#
# OUTPUTS:
# - sentiment label
# - confidence score
#
# DESIGN PRINCIPLES:
# - Reproducibility first
# - No hidden assumptions
# - Metadata-driven reconstruction
# - Future compatibility with:
#     * FastAPI
#     * Docker
#     * Airflow
#     * AWS SageMaker
#     * Azure ML
#     * Vertex AI
# ============================================================

import json
import pickle

import torch
import torch.nn.functional as F

from src.features.tokenizer import SimpleTokenizer
from src.models.model_factory import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_champion():
    """
    Load champion metadata.

    Returns
    -------
    dict
        Champion information.
    """

    with open("artifacts/champion.json", "r") as f:
        return json.load(f)


def load_metadata(model_name):
    """
    Load model metadata.

    Parameters
    ----------
    model_name : str

    Returns
    -------
    dict
    """

    path = f"artifacts/{model_name}_metadata.pkl"

    with open(path, "rb") as f:
        return pickle.load(f)


def load_vocab():
    """
    Load serialized vocabulary.

    Returns
    -------
    Vocabulary
    """

    with open("artifacts/vocab.pkl", "rb") as f:
        return pickle.load(f)


def build_config(metadata):
    """
    Convert metadata into the structure expected
    by model_factory.py.
    """

    return {
        "model": {
            "type": metadata["model_type"],
            "vocab_size": metadata["vocab_size"],
            "embed_dim": metadata["embed_dim"],
            "hidden_dim": metadata.get("hidden_dim", 128),
            "num_classes": metadata["num_classes"],
        }
    }


def load_model():
    """
    Fully reconstruct the champion model.

    Returns
    -------
    tuple
        (model, vocab)
    """

    champion = load_champion()

    model_name = champion["model"]

    metadata = load_metadata(model_name)

    config = build_config(metadata)

    model = get_model(config)

    model_path = f"models/{model_name}.pt"

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)

    model.eval()

    vocab = load_vocab()

    return model, vocab


# ============================================================
# LOAD ONCE AT STARTUP
# ============================================================

MODEL, VOCAB = load_model()

TOKENIZER = SimpleTokenizer()


def preprocess_text(text):
    """
    Convert raw text into a tensor.

    Parameters
    ----------
    text : str

    Returns
    -------
    torch.Tensor
    """

    tokens = TOKENIZER.tokenize(text)

    token_ids = [
        VOCAB.get(token)
        for token in tokens
    ]

    if len(token_ids) == 0:
        token_ids = [VOCAB.get(VOCAB.unk_token)]

    tensor = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


def predict_text(text):
    """
    Predict sentiment for a single text.

    Parameters
    ----------
    text : str

    Returns
    -------
    dict
        {
            "prediction": str,
            "confidence": float
        }
    """

    inputs = preprocess_text(text)

    with torch.no_grad():

        logits = MODEL(inputs)

        probabilities = F.softmax(
            logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    prediction_id = prediction.item()

    sentiment = (
        "negative"
        if prediction_id == 0
        else "positive"
    )

    return {
        "prediction": sentiment,
        "confidence": round(confidence.item(), 4),
    }


if __name__ == "__main__":

    examples = [
        "This movie was fantastic and I loved every minute.",
        "The film was boring and a complete waste of time.",
    ]

    for text in examples:

        result = predict_text(text)

        print("\nTEXT:")
        print(text)

        print("PREDICTION:")
        print(result)