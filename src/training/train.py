# ============================================================
# NLP TRAINING PIPELINE
# ============================================================

# ============================================================
# NLP TRAINING PIPELINE (MULTI-MODEL + MLFLOW + ARTIFACTS)
# ------------------------------------------------------------
# PURPOSE:
# Train, evaluate, compare, and persist multiple sentiment
# analysis models using the IMDb dataset.
#
# SUPPORTED ARCHITECTURES:
# - Baseline (Embedding + Mean Pooling)
# - LSTM
# - GRU
# - CNN (TextCNN)
# - Bidirectional LSTM
#
# MAIN RESPONSIBILITIES:
#
# 1) VOCABULARY CREATION
# ------------------------------------------------------------
# - Build the vocabulary from the IMDb training split.
# - Apply frequency filtering.
# - Save the vocabulary for future inference and deployment.
#
# GENERATED:
# artifacts/vocab.pkl
#
#
# 2) DATA PREPARATION
# ------------------------------------------------------------
# - Create training and testing datasets.
# - Tokenize text samples.
# - Convert tokens into integer IDs.
# - Dynamically pad sequences inside the collate function.
#
#
# 3) MODEL TRAINING
# ------------------------------------------------------------
# - Train multiple architectures sequentially.
# - Use a shared configuration object.
# - Use Adam optimization.
# - Compute training and testing metrics at every epoch.
#
#
# 4) MODEL EVALUATION
# ------------------------------------------------------------
# Metrics:
# - Loss
# - Accuracy
# - Precision
# - Recall
# - F1 Score
#
# PRIMARY RANKING METRIC:
# F1 Score (PRIMARY_METRIC = "f1")
#
#
# 5) EXPERIMENT TRACKING
# ------------------------------------------------------------
# MLflow is used to:
# - Track experiments.
# - Store parameters.
# - Store per-epoch metrics.
# - Compare architectures.
# - Prepare champion model selection.
#
# Tracking backend:
# sqlite:///mlflow.db
#
# Experiment:
# sentiment_experiment
#
#
# 6) ARTIFACT PERSISTENCE
# ------------------------------------------------------------
# Save model weights:
#
# models/
# ├── baseline.pt
# ├── lstm.pt
# ├── gru.pt
# ├── cnn.pt
# └── bilstm.pt
#
#
# Save metadata:
#
# artifacts/
# ├── baseline_metadata.pkl
# ├── lstm_metadata.pkl
# ├── gru_metadata.pkl
# ├── cnn_metadata.pkl
# ├── bilstm_metadata.pkl
# └── metrics.json
#
#
# 7) MODEL RECOVERABILITY
# ------------------------------------------------------------
# The pipeline preserves everything required to reproduce
# experiments:
#
# - Model weights (.pt)
# - Vocabulary (vocab.pkl)
# - Hyperparameters
# - Evaluation metrics
# - MLflow experiment history
#
# This avoids the reproducibility problems encountered in
# previous projects.
#
#
# 8) FUTURE INTEGRATIONS
# ------------------------------------------------------------
# This module is designed to support:
#
# - Champion model selection
# - FastAPI inference endpoints
# - GitHub Actions (CI/CD)
# - Streamlit dashboards
# - Gradio interfaces
# - Docker containerization
# - Airflow orchestration
# - DVC data versioning
# - AWS migration
# - Azure migration
# - GCP migration
# - Full cloud-native MLOps architectures
#
#
# DESIGN PRINCIPLES:
# ------------------------------------------------------------
# - Reproducibility
# - Experiment traceability
# - Artifact persistence
# - Recoverability
# - Extensibility
# - Cloud readiness
# ============================================================

import json
import os
import pickle
import warnings
from typing import Any

import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.config.constants import PRIMARY_METRIC
from src.data.dataset import SentimentDataset
from src.features.build_vocab import build_vocabulary_from_csv
from src.features.tokenizer import SimpleTokenizer
from src.models.model_factory import get_model

warnings.filterwarnings("ignore")


def collate_fn(batch):

    texts = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    max_len = max(len(text) for text in texts)

    padded_texts = []

    for text in texts:

        pad = torch.zeros(
            max_len - len(text),
            dtype=torch.long,
        )

        padded_texts.append(torch.cat([text, pad]))

    return {
        "text": torch.stack(padded_texts),
        "label": labels,
    }


def evaluate(model, dataloader):

    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():

        for batch in dataloader:

            x = batch["text"]
            y = batch["label"]

            outputs = model(x)

            loss = loss_fn(outputs, y)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            targets.extend(y.cpu().numpy())

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
    }


def save_model(model, model_name):

    os.makedirs("models", exist_ok=True)

    model_path = f"models/{model_name}.pt"

    torch.save(model.state_dict(), model_path)

    print(f"Saved model -> {model_path}")


def save_metadata(model_name, config, metrics):

    os.makedirs("artifacts", exist_ok=True)

    metadata = {
        "model_type": model_name,
        "vocab_size": config["model"]["vocab_size"],
        "embed_dim": config["model"]["embed_dim"],
        "hidden_dim": config["model"]["hidden_dim"],
        "num_classes": config["model"]["num_classes"],
        "batch_size": config["training"]["batch_size"],
        "epochs": config["training"]["epochs"],
        "learning_rate": config["training"]["lr"],
        "primary_metric": PRIMARY_METRIC,
        **metrics,
    }

    path = f"artifacts/{model_name}_metadata.pkl"

    with open(path, "wb") as file:
        pickle.dump(metadata, file)

    print(f"Saved metadata -> {path}")


def train_one_model(model_name, config, train_loader, test_loader):

    print("\\n==============================")
    print(f"Training model: {model_name}")
    print("==============================")

    config["model"]["type"] = model_name

    model = get_model(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("primary_metric", PRIMARY_METRIC)

        epochs = config["training"]["epochs"]

        for epoch in range(epochs):

            model.train()

            losses = []
            predictions = []
            targets = []

            for batch in train_loader:

                x = batch["text"]
                y = batch["label"]

                outputs = model(x)

                loss = loss_fn(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                preds = torch.argmax(outputs, dim=1)

                predictions.extend(preds.cpu().numpy())
                targets.extend(y.cpu().numpy())

            train_loss = np.mean(losses)

            train_accuracy = accuracy_score(
                targets,
                predictions,
            )

            test_metrics = evaluate(model, test_loader)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_accuracy:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} | "
                f"test_acc={test_metrics['accuracy']:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("test_loss", test_metrics["loss"], step=epoch)
            mlflow.log_metric("test_accuracy", test_metrics["accuracy"], step=epoch)

        final_metrics = evaluate(model, test_loader)

        for metric_name, metric_value in final_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    save_model(model, model_name)
    save_metadata(model_name, config, final_metrics)

    return final_metrics


def main():

    os.makedirs("models", exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sentiment_experiment")

    print("Config loaded")

    # ============================================================
    # VOCAB GENERATION (DVC-COMPLIANT)
    # ------------------------------------------------------------
    # Always rebuild vocabulary inside pipeline
    # ============================================================

    vocab = build_vocabulary_from_csv(split="train")

    os.makedirs("artifacts", exist_ok=True)

    # Save vocab as DVC artifact
    with open("artifacts/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print("Saved vocabulary -> artifacts/vocab.pkl")

    tokenizer = SimpleTokenizer()

    # ============================================================
    # MYPY COMPATIBILITY
    # ------------------------------------------------------------
    # Explicit typing prevents nested dictionaries from being
    # inferred as generic objects, allowing safe indexing like:
    #
    # config["training"]["batch_size"]
    # ============================================================

    config: dict[str, dict[str, Any]] = {
        "model": {
            "type": "baseline",
            "vocab_size": len(vocab.token_to_id),
            "embed_dim": 128,
            "hidden_dim": 128,
            "num_classes": 2,
        },
        "training": {
            "lr": 0.001,
            "batch_size": 8,
            "epochs": 1,
        },
    }

    train_dataset = SentimentDataset(
        "train",
        tokenizer,
        vocab,
    )

    test_dataset = SentimentDataset(
        "test",
        tokenizer,
        vocab,
    )

    print("FULL TRAIN SIZE:", len(train_dataset))
    print("FULL TEST SIZE:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    models_to_train = [
        "baseline",
        "lstm",
        "gru",
        "cnn",
        "bilstm",
    ]

    global_metrics = {}

    for model_name in models_to_train:

        metrics = train_one_model(
            model_name,
            config,
            train_loader,
            test_loader,
        )

        global_metrics[model_name] = metrics

    with open("artifacts/metrics.json", "w") as file:
        json.dump(global_metrics, file, indent=4)

    print("Saved metrics -> artifacts/metrics.json")


if __name__ == "__main__":
    main()
