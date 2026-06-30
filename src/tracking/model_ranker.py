# ============================================================
# MODEL RANKER (PRODUCTION-READY VERSION WITH CHAMPION LOCKING)
# ------------------------------------------------------------
# PURPOSE:
# Loads MLflow experiments and produces a clean leaderboard.
# Automatically selects and persists the champion model.
#
# FEATURES:
# - Reads MLflow runs
# - Cleans invalid runs
# - Ranks models using PRIMARY_METRIC (F1)
# - Persists champion model to artifacts/champion.json
# - Ensures reproducibility for inference layer
#
# CRITICAL DESIGN PRINCIPLE:
# Champion selection MUST be deterministic and persisted.
#
# MYPY NOTES:
# - MLflow's search_runs() type hints are incomplete.
# - We explicitly annotate the returned object as a
#   Pandas DataFrame to enable static analysis.
# ============================================================

import json
import os

import mlflow
import pandas as pd

from src.config.constants import PRIMARY_METRIC

CHAMPION_PATH = "artifacts/champion.json"


def save_champion(row: pd.Series) -> None:
    """
    Persist champion model metadata for inference use.
    """

    os.makedirs("artifacts", exist_ok=True)

    champion_data = {
        "model": row["model"],
        "run_id": row["run_id"],
        "metric": PRIMARY_METRIC,
        "score": float(row[PRIMARY_METRIC]),
    }

    with open(CHAMPION_PATH, "w") as f:
        json.dump(champion_data, f, indent=4)

    print(f"\nChampion saved to {CHAMPION_PATH}")


def main() -> None:

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    experiment = mlflow.get_experiment_by_name("sentiment_experiment")

    if experiment is None:
        print("Experiment not found.")
        return

    print("\nLoading MLflow runs...\n")

    # ========================================================
    # LOAD MLFLOW RUNS
    # --------------------------------------------------------
    # Explicit annotation required because MLflow exposes
    # incomplete type information to MyPy.
    # ========================================================
    runs: pd.DataFrame = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    columns = [
        "run_id",
        "params.model_type",
        "metrics.test_accuracy",
        "metrics.f1",
        "metrics.precision",
        "metrics.recall",
        "metrics.test_loss",
    ]

    # Keep only available columns safely
    available_columns = runs.columns.tolist()
    valid_columns = [c for c in columns if c in available_columns]

    runs = runs[valid_columns]

    runs.columns = [
        "run_id",
        "model",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "loss",
    ]

    print("\nRAW DATA:")
    print(runs)

    # ========================================================
    # CLEANING
    # ========================================================
    clean_df = runs.copy()

    clean_df = clean_df[(clean_df["accuracy"] > 0.0) & (clean_df["accuracy"] < 1.0)]

    # ========================================================
    # RANKING
    # ========================================================
    clean_df = clean_df.sort_values(PRIMARY_METRIC, ascending=False)

    clean_df = clean_df.groupby("model", as_index=False).first()

    leaderboard = clean_df.sort_values(
        PRIMARY_METRIC,
        ascending=False,
    ).reset_index(drop=True)

    leaderboard["rank"] = leaderboard.index + 1

    print("\nFINAL LEADERBOARD:")
    print(leaderboard)

    best_model = leaderboard.iloc[0]

    print("\nBEST MODEL:")
    print(best_model)

    # ========================================================
    # CHAMPION LOCKING
    # ========================================================
    save_champion(best_model)


if __name__ == "__main__":
    main()
