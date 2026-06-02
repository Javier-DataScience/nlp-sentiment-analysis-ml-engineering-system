"""
train.py

PURPOSE:
- Immutable training entrypoint for all experiments
- Runs any model defined in a YAML config
- Ensures reproducibility and MLflow tracking consistency

ARCHITECTURE RULES:
- NO model definitions here
- NO MLflow logic here (only calls tracker)
- NO hardcoded hyperparameters
- EVERYTHING comes from config
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import SentimentDataset
from src.features.collate import collate_batch

from src.models.model_factory import get_model
from src.training.trainer import Trainer
from src.tracking.mlflow_tracker import MLflowTracker


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str):

    # -------------------------
    # 1. LOAD CONFIG (SOURCE OF TRUTH)
    # -------------------------
    config = load_config(config_path)

    # -------------------------
    # 2. DATASET + DATALOADER
    # -------------------------
    dataset = SentimentDataset(
        path=config["data"]["train_path"],
        tokenizer=None,   # already embedded in dataset design
        vocab=None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_batch
    )

    # -------------------------
    # 3. MODEL (MODEL ZOO)
    # -------------------------
    model = get_model(config["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -------------------------
    # 4. LOSS + OPTIMIZER
    # -------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"]
    )

    # -------------------------
    # 5. MLFLOW TRACKER (CONTROLLED LAYER)
    # -------------------------
    tracker = MLflowTracker(
        experiment_name=config["experiment_name"]
    )

    tracker.start_run(config)

    # -------------------------
    # 6. TRAINING LOOP
    # -------------------------
    for epoch in range(config["training"]["epochs"]):

        epoch_loss = 0.0

        for batch in dataloader:
            inputs = batch["text"].to(device)
            labels = batch["labels"].float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # -------------------------
        # LOGGING (CENTRALIZED)
        # -------------------------
        tracker.log_metric("loss", avg_loss, epoch)

    # -------------------------
    # 7. SAVE MODEL ARTIFACT
    # -------------------------
    tracker.save_model(model)

    tracker.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    main(args.config)