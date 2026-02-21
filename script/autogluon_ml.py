#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
autogluon_ml.py

AutoGluon-based multivariate regression for predicting chlorophyll-a
concentrations from ancillary oceanographic variables.

Steps
-----
1. Load the dataset and drop the ``Date`` column.
2. Split into 80/20 train/test (random, seed=128).
3. Train an AutoGluon TabularPredictor ensemble.
4. Evaluate on the held-out test set (RMSE, R^2, MAE, etc.).
5. Compute and save feature importances.
6. Refit the best model on the full dataset for deployment.

Outputs (data)
--------------
- ../processed_data/devel_leaderboard.csv    : Model leaderboard.
- ../processed_data/devel_features_imp.csv   : Feature importance table.

Outputs (models)
----------------
- ../autogluon_models/trained_models/        : Development artefacts.
- ../autogluon_models/final_model_for_deployment/ : Deployment-ready model.

Outputs (report)
----------------
- ../reports/autogluon_ml.txt : Training and evaluation summary.

Author  : Sandy Herho <sh001@ucr.edu>
Date    : 2026/02/21
License : MIT
"""

import os
import warnings
from datetime import datetime

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("..", "raw_data", "all_data.csv")
PROCESSED_DIR = os.path.join("..", "processed_data")
MODEL_DIR = os.path.join("..", "autogluon_models", "trained_models")
DEPLOY_DIR = os.path.join("..", "autogluon_models", "final_model_for_deployment")
REPORT_PATH = os.path.join("..", "reports", "autogluon_ml.txt")
LABEL = "Chlorophyll_a"
TRAIN_FRAC = 0.8
RANDOM_STATE = 128


# ===========================================================================
# AutoGluonWorkflow
# ===========================================================================
class AutoGluonWorkflow:
    """End-to-end AutoGluon tabular regression workflow.

    Parameters
    ----------
    data_path : str
        Path to the input CSV file.
    save_path : str
        Directory for trained-model artefacts.
    """

    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.predictor = None

    def load_and_prepare_data(self):
        """Load CSV and drop the ``Date`` column.

        Returns
        -------
        data : pd.DataFrame or None
        """
        try:
            data = TabularDataset(self.data_path)
            data.drop(columns="Date", inplace=True)
            return data
        except Exception as exc:
            print(f"[ERROR] Failed to load data: {exc}")
            return None

    def split_data(self, data):
        """Random 80/20 train-test split.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        train_data, test_data : pd.DataFrame
        """
        try:
            train_size = round(TRAIN_FRAC * len(data))
            train_data = data.sample(train_size, random_state=RANDOM_STATE)
            test_data = data.drop(train_data.index)
            return train_data, test_data
        except Exception as exc:
            print(f"[ERROR] Failed to split data: {exc}")
            return None, None

    def train_and_evaluate(self, train_data, test_data):
        """Train the AutoGluon ensemble and evaluate on the test set.

        Parameters
        ----------
        train_data, test_data : pd.DataFrame

        Returns
        -------
        leaderboard : pd.DataFrame
        metrics : dict
        feature_importance : pd.DataFrame
        """
        # Train
        self.predictor = TabularPredictor(
            label=LABEL, path=self.save_path
        ).fit(train_data)

        # Leaderboard
        leaderboard = self.predictor.leaderboard()

        # Evaluate
        y_test = test_data[LABEL]
        test_features = test_data.drop(columns=[LABEL])
        y_pred = self.predictor.predict(test_features)
        metrics = self.predictor.evaluate_predictions(
            y_true=y_test, y_pred=y_pred, auxiliary_metrics=True
        )

        # Feature importance
        feature_importance = self.predictor.feature_importance(test_data)

        return leaderboard, metrics, feature_importance

    def get_predictor(self):
        """Return the trained ``TabularPredictor`` instance."""
        return self.predictor


# ===========================================================================
# Deployment helper
# ===========================================================================
def deploy_model(model_dir, deploy_dir):
    """Refit the best model on all data and clone for deployment.

    Parameters
    ----------
    model_dir : str
        Path to the development model artefacts.
    deploy_dir : str
        Path for the deployment-ready artefact.
    """
    predictor = TabularPredictor.load(model_dir)
    predictor.refit_full()
    predictor.clone_for_deployment(deploy_dir)
    print(f"[INFO] Deployment model saved -> {deploy_dir}")


# ===========================================================================
# Report
# ===========================================================================
def write_report(filepath, leaderboard, metrics, feature_importance,
                 n_train, n_test, n_total):
    """Write a training / evaluation summary report.

    Parameters
    ----------
    filepath : str
        Output text file path.
    leaderboard : pd.DataFrame
    metrics : dict
    feature_importance : pd.DataFrame
    n_train, n_test, n_total : int
        Sample counts.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sep = "=" * 60
    lines = [
        sep,
        "AutoGluon Multivariate Regression Report",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
        "",
        "--- Dataset Split ---",
        f"  Total samples   : {n_total}",
        f"  Training samples: {n_train} ({TRAIN_FRAC * 100:.0f}%)",
        f"  Test samples    : {n_test} ({(1 - TRAIN_FRAC) * 100:.0f}%)",
        f"  Random state    : {RANDOM_STATE}",
        f"  Target variable : {LABEL}",
        "",
        "--- Model Leaderboard ---",
        leaderboard.to_string(index=False),
        "",
        "--- Test-Set Evaluation Metrics ---",
    ]
    for key, value in metrics.items():
        lines.append(f"  {key:30s}: {value:.4f}")

    lines.extend([
        "",
        "--- Feature Importance (permutation-based) ---",
        feature_importance.to_string(),
        "",
        "--- Interpretation ---",
        "AutoGluon trained an ensemble of models and automatically",
        "selected the best configuration. Evaluation metrics above",
        "reflect out-of-sample (test set) performance. Feature",
        "importances are computed via permutation on the test set;",
        "higher values indicate greater predictive relevance.",
        "",
        f"Deployment model saved to: {DEPLOY_DIR}",
        sep,
    ])
    with open(filepath, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[INFO] Report saved -> {filepath}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    """Run the full AutoGluon pipeline."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load and prepare
    workflow = AutoGluonWorkflow(DATA_PATH, MODEL_DIR)
    data = workflow.load_and_prepare_data()
    if data is None:
        return

    # 2. Split
    train_data, test_data = workflow.split_data(data)
    if train_data is None:
        return

    n_total = len(data)
    n_train = len(train_data)
    n_test = len(test_data)
    print(f"[INFO] Train: {n_train}, Test: {n_test}, Total: {n_total}")

    # 3. Train and evaluate
    leaderboard, metrics, feature_importance = workflow.train_and_evaluate(
        train_data, test_data
    )

    # 4. Save artefacts
    lb_path = os.path.join(PROCESSED_DIR, "devel_leaderboard.csv")
    leaderboard.to_csv(lb_path, index=False)
    print(f"[INFO] Saved -> {lb_path}")

    fi_path = os.path.join(PROCESSED_DIR, "devel_features_imp.csv")
    feature_importance.to_csv(fi_path)
    print(f"[INFO] Saved -> {fi_path}")

    # 5. Print metrics
    print("\n--- Evaluation Metrics ---")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")

    print("\n--- Feature Importance ---")
    print(feature_importance)

    # 6. Deploy
    deploy_model(MODEL_DIR, DEPLOY_DIR)

    # 7. Report
    write_report(
        REPORT_PATH, leaderboard, metrics, feature_importance,
        n_train, n_test, n_total
    )


if __name__ == "__main__":
    main()
