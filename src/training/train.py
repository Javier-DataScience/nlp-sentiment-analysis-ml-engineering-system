# ============================================================
# NLP TRAINING PIPELINE (IMDB + MULTI-MODEL + MLFLOW)
# ------------------------------------------------------------
# PURPOSE:
# This script trains multiple NLP models (baseline, LSTM, GRU)
# on the IMDb dataset with proper vocabulary handling and
# MLflow experiment tracking.
#
# FIXES INCLUDED (STEP 10):
# - Correct vocab initialization order (prevents UnboundLocalError)
# - Stable vocab size injection into model config
# - Safer evaluation (CPU conversion fixes)
# - Cleaner reproducible training loop
# - Consistent experiment tracking across models
#
# OUTPUT:
# - Per-epoch logs (loss, accuracy)
# - MLflow tracking per model run
# - Final precision/recall/F1 per model
# ============================================================

import torch
import numpy as np
import mlflow
import warnings

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.features.tokenizer import SimpleTokenizer
from src.features.vocabulary import Vocabulary
from src.features.build_vocab import build_vocabulary_from_csv
from src.data.dataset import SentimentDataset
from src.models.model_factory import get_model


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
# EVALUATION FUNCTION (STABLE VERSION)
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

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

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
    config["model"]["vocab_size"] = config["model"]["vocab_size"]

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

                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(y.cpu().numpy())

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

    print("Config loaded")

    # ========================================================
    # STEP 1: BUILD VOCAB FIRST (CRITICAL FIX)
    # ========================================================
    print("Building vocabulary from IMDb (train)...")

    vocab = build_vocabulary_from_csv("data/raw/train.csv")

    print("Vocabulary built successfully")
    print("Vocab size:", len(vocab.token_to_id))

    # ========================================================
    # STEP 2: TOKENIZER
    # ========================================================
    tokenizer = SimpleTokenizer()

    # ========================================================
    # STEP 3: CONFIG (NOW SAFE)
    # ========================================================
    config = {
        "model": {
            "type": "baseline",
            "vocab_size": len(vocab.token_to_id),
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

    # ========================================================
    # DATASETS
    # ========================================================
    train_dataset = SentimentDataset("train", tokenizer, vocab)
    test_dataset = SentimentDataset("test", tokenizer, vocab)

    print("FULL TRAIN SIZE:", len(train_dataset))
    print("FULL TEST SIZE:", len(test_dataset))

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

    # ========================================================
    # MULTI MODEL TRAINING
    # ========================================================
    models_to_train = ["baseline", "lstm", "gru"]

    for model_name in models_to_train:
        train_one_model(model_name, config, train_loader, test_loader)


if __name__ == "__main__":
    main()