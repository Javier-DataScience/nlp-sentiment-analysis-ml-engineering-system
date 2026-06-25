# ============================================================
# MODEL RANKER (PRODUCTION-READY VERSION)
# ------------------------------------------------------------
# PURPOSE:
# Loads MLflow experiments and produces a clean leaderboard.
#
# FEATURES:
# - Reads all runs from the sentiment experiment
# - Removes invalid experiments (accuracy = 0 or accuracy = 1)
# - Keeps only the best run per model
# - Uses a centralized PRIMARY_METRIC from config/constants.py
# - Produces a final ranked leaderboard
#
# CURRENT PRIMARY METRIC:
# - F1 Score
#
# FUTURE:
# Changing the ranking metric only requires modifying:
#
#     src/config/constants.py
#
# Example:
#
#     PRIMARY_METRIC = "accuracy"
#     PRIMARY_METRIC = "recall"
#     PRIMARY_METRIC = "precision"
# ============================================================

import mlflow
import pandas as pd

from src.config.constants import PRIMARY_METRIC


def main():

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    experiment = mlflow.get_experiment_by_name(
        "sentiment_experiment"
    )

    if experiment is None:
        print("Experiment not found.")
        return

    print("\nLoading MLflow runs...\n")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    columns = [
        "run_id",
        "params.model",
        "metrics.test_accuracy",
        "metrics.f1",
        "metrics.precision",
        "metrics.recall",
        "metrics.test_loss"
    ]

    runs = runs[columns]

    runs.columns = [
        "run_id",
        "model",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "loss"
    ]

    print("\nRAW DATA:")
    print(runs)

    # ========================================================
    # REMOVE BROKEN / DIAGNOSTIC RUNS
    # ========================================================
    clean_df = runs.copy()

    clean_df = clean_df[
        (clean_df["accuracy"] > 0.0)
        & (clean_df["accuracy"] < 1.0)
    ]

    # ========================================================
    # KEEP BEST RUN PER MODEL
    # ========================================================
    clean_df = clean_df.sort_values(
        PRIMARY_METRIC,
        ascending=False
    )

    clean_df = clean_df.groupby(
        "model",
        as_index=False
    ).first()

    print("\nCLEAN DATA:")
    print(clean_df)

    # ========================================================
    # FINAL LEADERBOARD
    # ========================================================
    leaderboard = clean_df.sort_values(
        PRIMARY_METRIC,
        ascending=False
    ).reset_index(drop=True)

    leaderboard["rank"] = leaderboard.index + 1

    print(
        f"\nFINAL LEADERBOARD (RANKED BY {PRIMARY_METRIC.upper()}):"
    )
    print(leaderboard)

    print("\nBEST MODEL:")

    print(leaderboard.iloc[0])


if __name__ == "__main__":
    main()