# ============================================================
# MODEL RANKING SYSTEM (PRODUCTION STYLE - MLFLOW)
# ------------------------------------------------------------
# Purpose:
# This module reads MLflow experiment runs and produces a
# clean, de-duplicated model leaderboard.
#
# It is designed for:
# - model comparison
# - best model selection
# - MLOps-ready evaluation logic
#
# Key features:
# - removes duplicate runs
# - groups by model name
# - aggregates metrics
# - ranks using F1 score (primary metric)
# ============================================================

import mlflow
import pandas as pd


# ============================================================
# LOAD ALL RUNS FROM EXPERIMENT
# ============================================================
def load_runs(experiment_name="sentiment_experiment"):

    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(experiment.experiment_id)

    rows = []

    for run in runs:

        data = run.data.metrics

        rows.append({
            "run_id": run.info.run_id,
            "model": run.data.tags.get("mlflow.runName", "unknown"),
            "accuracy": data.get("test_accuracy"),
            "f1": data.get("f1"),
            "precision": data.get("precision"),
            "recall": data.get("recall"),
            "loss": data.get("test_loss"),
        })

    return pd.DataFrame(rows)


# ============================================================
# CLEAN DUPLICATES + INVALID RUNS
# ============================================================
def clean_data(df):

    # Remove rows with missing critical metrics
    df = df.dropna(subset=["f1", "accuracy"])

    # Keep only best run per model (by F1)
    df = df.sort_values("f1", ascending=False)
    df = df.drop_duplicates(subset=["model"], keep="first")

    return df


# ============================================================
# RANK MODELS
# ============================================================
def rank_models(df):

    df = df.sort_values(by="f1", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


# ============================================================
# BEST MODEL SELECTOR
# ============================================================
def get_best_model(df):

    return df.iloc[0]


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    print("\nLoading MLflow runs...")

    df = load_runs()

    print("\nRAW DATA:")
    print(df)

    df = clean_data(df)

    print("\nCLEAN DATA (DEDUP + FILTERED):")
    print(df)

    ranked = rank_models(df)

    print("\nFINAL LEADERBOARD (RANKED BY F1):")
    print(ranked)

    best = get_best_model(ranked)

    print("\nBEST MODEL:")
    print(best)