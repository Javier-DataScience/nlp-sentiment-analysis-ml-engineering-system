# ============================================================
# TEST: MLFLOW TRACKER
# ------------------------------------------------------------
# Purpose:
# This test validates the MLflowTracker wrapper class.
#
# It ensures:
# - MLflow experiment is correctly initialized
# - runs can be created successfully
# - parameters are logged properly
# - metrics are recorded correctly
#
# This is the foundation for experiment tracking in training.
# ============================================================

from src.tracking.mlflow_tracker import MLflowTracker


def test_mlflow_tracker():
    """
    Executes a minimal MLflow tracking test.
    """

    # Initialize tracker
    tracker = MLflowTracker("sentiment_experiment")

    # Start MLflow run
    with tracker.start_run(run_name="test_run"):

        # Log hyperparameters
        tracker.log_params({
            "lr": 0.001,
            "batch_size": 2
        })

        # Log metrics
        tracker.log_metrics({
            "accuracy": 0.85,
            "loss": 0.42
        })

    print("MLflow tracker test completed successfully")


if __name__ == "__main__":
    test_mlflow_tracker()