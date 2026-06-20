# ============================================================
# MODEL EVALUATION & RANKING SYSTEM (MLFLOW ANALYSIS)
# ------------------------------------------------------------
# Purpose:
# This module extracts MLflow experiment runs and builds
# a comparison table of all trained models.
#
# It allows:
# - Ranking models by metrics (accuracy, f1, loss)
# - Identifying best model
# - Comparing training performance
#
# Works with:
# - sentiment_experiment (MLflow)
# ============================================================

import mlflow
import pandas as pd


# ============================================================
# LOAD EXPERIMENT RUNS
# ============================================================
def load_experiment_runs(experiment_name="sentiment_experiment"):

    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    runs = client.search_runs(experiment.experiment_id)

    data = []

    for run in runs:

        metrics = run.data.metrics

        data.append({
            "run_id": run.info.run_id,
            "model": run.data.tags.get("mlflow.runName", "unknown"),
            "accuracy": metrics.get("test_accuracy", None),
            "f1": metrics.get("f1", None),
            "precision": metrics.get("precision", None),
            "recall": metrics.get("recall", None),
            "loss": metrics.get("test_loss", None),
        })

    return pd.DataFrame(data)


# ============================================================
# RANK MODELS
# ============================================================
def rank_models(df, metric="accuracy"):

    df = df.dropna(subset=[metric])

    df = df.sort_values(by=metric, ascending=False)

    df["rank"] = range(1, len(df) + 1)

    return df


# ============================================================
# BEST MODEL SELECTOR
# ============================================================
def get_best_model(df, metric="accuracy"):

    df = df.dropna(subset=[metric])

    best = df.sort_values(by=metric, ascending=False).iloc[0]

    return best


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    df = load_experiment_runs()

    print("\n=== RAW EXPERIMENT DATA ===")
    print(df)

    print("\n=== RANKED BY ACCURACY ===")
    ranked = rank_models(df, metric="accuracy")
    print(ranked)

    best = get_best_model(df, metric="accuracy")

    print("\n=== BEST MODEL ===")
    print(best)