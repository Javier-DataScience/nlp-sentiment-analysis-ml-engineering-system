"""
train.py

============================================================
PURPOSE:
------------------------------------------------------------
This is the core training pipeline of the NLP system.

It is designed to be IMMUTABLE:
- No model definitions here
- No hyperparameters hardcoded
- No experiment logic mixed in

Everything is driven by external configuration (YAML).

This ensures:
- reproducibility
- MLflow consistency
- cloud compatibility (SageMaker / Azure ML)
- clean separation of concerns

============================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.model_factory import get_model
from src.data.dataset import get_dataloaders
from src.utils.config import load_config
from src.tracking.mlflow_tracker import MLflowTracker


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train(config_path: str):

    # --------------------------------------------------------
    # 1. LOAD CONFIG
    # --------------------------------------------------------
    config = load_config(config_path)

    # --------------------------------------------------------
    # 2. DEVICE SETUP
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # 3. DATA
    # --------------------------------------------------------
    train_loader, val_loader = get_dataloaders(config)

    # --------------------------------------------------------
    # 4. MODEL (via factory)
    # --------------------------------------------------------
    model = get_model(config)
    model.to(device)

    # --------------------------------------------------------
    # 5. LOSS + OPTIMIZER
    # --------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"]
    )

    # --------------------------------------------------------
    # 6. MLflow tracking (centralized wrapper)
    # --------------------------------------------------------
    tracker = MLflowTracker(config)
    tracker.start_run()

    # --------------------------------------------------------
    # 7. TRAINING LOOP
    # --------------------------------------------------------
    epochs = config["training"]["epochs"]

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for batch in train_loader:

            inputs = batch["text"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------
        tracker.log_metric("train_loss", avg_loss, step=epoch)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # --------------------------------------------------------
    # 8. SAVE MODEL ARTIFACT
    # --------------------------------------------------------
    model_path = config["paths"]["model_output"]

    torch.save(model.state_dict(), model_path)

    tracker.log_model(model, model_path)

    tracker.end_run()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    train(args.config)