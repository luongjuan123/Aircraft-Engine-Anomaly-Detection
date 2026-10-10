#!/usr/bin/env python3
"""
Anomaly Detection on Combined N-CMAPSS Dataset (Folder 17) using Gaussian Mixture Model (EM).

Workflow:
1. Load train.csv, dev.csv, and test.csv from Processed data/.
2. Select non-constant physical sensor and operational flight condition features.
3. Standardize features based on normal operating cycles (RUL > 20 cycles).
4. Train GaussianMixture model with EM.
5. Calibrate log-likelihood anomaly threshold on dev.csv.
6. Evaluate anomaly detection performance on test.csv.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from gmm import GaussianMixture


def prepare_features(df_train, df_dev, df_test, rul_threshold=20):
    exclude_cols = {"unit", "cycle", "Fc", "hs", "RUL", "source", "anomaly", "dataset_id"}
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    stds = df_train[feature_cols].std()
    active_features = stds[stds > 1e-4].index.tolist()
    print(f"Selected {len(active_features)} active features out of {len(feature_cols)}.")

    for df in [df_train, df_dev, df_test]:
        df["anomaly"] = (df["RUL"] <= rul_threshold).astype(int)

    train_normal = df_train[df_train["anomaly"] == 0]

    mean = train_normal[active_features].mean().values
    std = train_normal[active_features].std().values
    std = np.where(std < 1e-6, 1.0, std)

    X_train_normal = (train_normal[active_features].values - mean) / std
    X_dev = (df_dev[active_features].values - mean) / std
    X_test = (df_test[active_features].values - mean) / std

    y_dev = df_dev["anomaly"].values
    y_test = df_test["anomaly"].values

    return X_train_normal, X_dev, y_dev, X_test, y_test, active_features


def compute_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


def find_best_threshold(log_probs, y_true):
    percentiles = np.linspace(1, 99, 100)
    candidate_thresholds = np.percentile(log_probs, percentiles)

    best_thresh = candidate_thresholds[0]
    best_f1 = -1.0
    best_metrics = None

    for thresh in candidate_thresholds:
        y_pred = (log_probs < thresh).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thresh = thresh
            best_metrics = metrics

    return best_thresh, best_metrics


def run_anomaly_detection(k=6, max_iter=50, reg_cov=1e-3, seed=42):
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "Processed data"

    print("===============================================================")
    print("  GMM Anomaly Detection on Combined Folder 17 Dataset")
    print("===============================================================")
    print(f"Loading data from {data_dir}...")

    # Load from the single train.csv, dev.csv, test.csv
    df_train = pd.read_csv(data_dir / "train.csv")
    df_dev = pd.read_csv(data_dir / "dev.csv")
    df_test = pd.read_csv(data_dir / "test.csv")

    X_train_normal, X_dev, y_dev, X_test, y_test, features = prepare_features(df_train, df_dev, df_test)

    # Subsample normal training set for fast GMM EM convergence
    max_train = 50000
    if len(X_train_normal) > max_train:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train_normal), size=max_train, replace=False)
        X_fit = X_train_normal[idx]
        print(f"Subsampled {max_train:,} normal cycles from {len(X_train_normal):,} for GMM training.")
    else:
        X_fit = X_train_normal

    print(f"Training GMM (k={k}, reg_cov={reg_cov}, max_iter={max_iter})...")
    gmm = GaussianMixture(k=k, max_iter=max_iter, reg_cov=reg_cov, random_state=seed)
    gmm.fit(X_fit)

    print(f"EM converged in {gmm.n_iter_} iterations.")
    print(f"Mixture weights phi: {np.round(gmm.phi_, 4)}")

    log_p_dev = gmm.score_samples(X_dev)
    log_p_test = gmm.score_samples(X_test)

    threshold, dev_metrics = find_best_threshold(log_p_dev, y_dev)
    print(f"\n[Dev Set Calibration]")
    print(f"Optimal log-likelihood threshold: {threshold:.4f}")
    print(f"Dev F1: {dev_metrics['f1']:.4f} | Precision: {dev_metrics['precision']:.4f} | Recall: {dev_metrics['recall']:.4f} | Accuracy: {dev_metrics['accuracy']:.4f}")

    y_test_pred = (log_p_test < threshold).astype(int)
    test_metrics = compute_metrics(y_test, y_test_pred)

    print(f"\n[Test Set Evaluation]")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1 Score:  {test_metrics['f1']:.4f}")
    print(f"Confusion Matrix: TP={test_metrics['tp']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TN={test_metrics['tn']}")

    return gmm, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMM Anomaly Detection on Folder 17")
    parser.add_argument("--k", type=int, default=6, help="Number of Gaussian components")
    parser.add_argument("--max_iter", type=int, default=50, help="Max EM iterations")
    parser.add_argument("--reg_cov", type=float, default=1e-3, help="Covariance regularization")
    args = parser.parse_args()

    run_anomaly_detection(k=args.k, max_iter=args.max_iter, reg_cov=args.reg_cov)
