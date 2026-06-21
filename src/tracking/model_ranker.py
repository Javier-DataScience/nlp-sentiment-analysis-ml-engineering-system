# ============================================================
# MODEL RANKER (SINGLE SOURCE OF TRUTH FOR MODEL SELECTION)
# ------------------------------------------------------------
# PURPOSE:
# This module defines how models are compared and selected.
#
# DESIGN PRINCIPLES:
# - One primary metric (F1-score)
# - Deterministic ranking
# - Clean MLflow integration
# - No duplicated logic across scripts
#
# OUTPUT:
# - Ranked leaderboard of all experiments
# - Best model selection
# ============================================================

import pandas as pd
import mlflow


# ============================================================
# CONFIGURATION (SINGLE SOURCE OF TRUTH)
# ============================================================
PRIMARY_METRIC = "f1"
SECONDARY_METRICS = ["accuracy", "loss"]


# ============================================================
# LOAD EXPERIMENTS FROM MLFLOW
# ============================================================
def load_experiments():

    print("\nLoading MLflow runs...\n")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("sentiment_experiment")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    data = []

    for run in runs:
        metrics = run.data.metrics

        data.append({
            "run_id": run.info.run_id,
            "model": run.data.tags.get("mlflow.runName", "unknown"),
            "accuracy": metrics.get("test_accuracy", 0),
            "f1": metrics.get("f1", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "loss": metrics.get("test_loss", 0),
        })

    return pd.DataFrame(data)


# ============================================================
# CLEAN + DEDUPLICATE RESULTS
# ============================================================
def clean_data(df):

    # keep best run per model (by PRIMARY_METRIC)
    df = df.sort_values(PRIMARY_METRIC, ascending=False)
    df = df.drop_duplicates(subset=["model"], keep="first")

    return df


# ============================================================
# RANK MODELS
# ============================================================
def rank_models(df):

    df = df.sort_values(PRIMARY_METRIC, ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


# ============================================================
# SELECT BEST MODEL
# ============================================================
def select_best_model(df):

    best = df.iloc[0]

    print("\nBEST MODEL:")
    print(best)

    return best


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    df = load_experiments()

    print("\nRAW DATA:")
    print(df)

    df = clean_data(df)

    print("\nCLEAN DATA:")
    print(df)

    df = rank_models(df)

    print("\nFINAL LEADERBOARD:")
    print(df)

    best = select_best_model(df)