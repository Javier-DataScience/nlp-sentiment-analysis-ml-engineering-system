# ============================================================
# MLflow Tracker (CLEAN RESET VERSION)
# ------------------------------------------------------------
# Purpose:
# This is a clean MLflow setup after full reset.
# It uses MLflow default tracking behavior (mlruns folder).
#
# Key idea:
# - No SQLite forcing
# - No file URI forcing
# - Let MLflow manage local tracking automatically
# ============================================================

import mlflow


class MLflowTracker:

    def __init__(self, experiment_name: str):
        # Use MLflow default local tracking (creates mlruns/)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str = None):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def log_metric(self, key: str, value: float):
        mlflow.log_metric(key, value)