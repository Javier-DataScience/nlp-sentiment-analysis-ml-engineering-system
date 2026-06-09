# ============================================================
# NLP TRAINING PIPELINE (IMDb + MULTI-MODEL + MLFLOW)
# ============================================================
# PURPOSE:
# This module trains multiple NLP models (baseline, LSTM)
# on the IMDb sentiment dataset from HuggingFace.
#
# It performs:
# - Data loading (IMDb train/test splits)
# - Tokenization + vocabulary encoding
# - Padding via custom collate function
# - Model training loop (multi-model support)
# - Evaluation per epoch (train + test metrics)
# - MLflow experiment tracking (loss, accuracy, F1, etc.)
#
# OUTPUT:
# - Terminal logs per epoch:
#     train_loss, train_accuracy, test_loss, test_accuracy
# - MLflow UI tracking:
#     all metrics per epoch + final precision/recall/F1
#
# MODELS TRAINED:
# - baseline
# - lstm
#
# NOTE:
# Each model is trained sequentially and logged as a
# separate MLflow run.
# ============================================================

import torch
import numpy as np
import mlflow
import warnings

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.features.tokenizer import SimpleTokenizer
from src.features.vocabulary import Vocabulary
from src.data.dataset import SentimentDataset
from src.models.model_factory import get_model


# ============================================================
# SUPPRESS WARNINGS (clean terminal output)
# ============================================================
warnings.filterwarnings("ignore")


# ============================================================
# COLLATE FUNCTION (PAD SEQUENCES)
# ============================================================
def collate_fn(batch):

    texts = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    max_len = max(len(t) for t in texts)

    padded_texts = []

    for t in texts:
        pad = torch.zeros(max_len - len(t), dtype=torch.long)
        padded_texts.append(torch.cat([t, pad]))

    return {
        "text": torch.stack(padded_texts),
        "label": labels
    }


# ============================================================
# MODEL EVALUATION
# ============================================================
def evaluate(model, dataloader):

    model.eval()

    all_preds = []
    all_labels = []

    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:

            x = batch["text"]
            y = batch["label"]

            outputs = model(x)
            loss = loss_fn(outputs, y)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }


# ============================================================
# TRAIN SINGLE MODEL
# ============================================================
def train_one_model(model_name, config, train_loader, test_loader):

    print("\n==============================")
    print(f"Training model: {model_name}")
    print("==============================")

    config["model"]["type"] = model_name
    model = get_model(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=model_name):

        epochs = config["training"]["epochs"]

        for epoch in range(epochs):

            model.train()

            losses = []
            preds_all = []
            labels_all = []

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

                preds_all.extend(preds.numpy())
                labels_all.extend(y.numpy())

            train_loss = np.mean(losses)
            train_acc = accuracy_score(labels_all, preds_all)

            test_metrics = evaluate(model, test_loader)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} | "
                f"test_acc={test_metrics['accuracy']:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("test_loss", test_metrics["loss"], step=epoch)
            mlflow.log_metric("test_accuracy", test_metrics["accuracy"], step=epoch)

        final = evaluate(model, test_loader)

        mlflow.log_metric("precision", final["precision"])
        mlflow.log_metric("recall", final["recall"])
        mlflow.log_metric("f1", final["f1"])


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main():

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sentiment_experiment")

    config = {
        "model": {
            "type": "baseline",
            "vocab_size": 30522,
            "embed_dim": 128,
            "hidden_dim": 128,
            "num_classes": 2
        },
        "training": {
            "lr": 0.001,
            "batch_size": 8,
            "epochs": 3
        }
    }

    print("Config loaded")

    tokenizer = SimpleTokenizer()
    vocab = Vocabulary()

    train_dataset = SentimentDataset("train", tokenizer, vocab)
    test_dataset = SentimentDataset("test", tokenizer, vocab)

    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    models_to_train = ["baseline", "lstm"]

    for model_name in models_to_train:
        train_one_model(model_name, config, train_loader, test_loader)


if __name__ == "__main__":
    main()