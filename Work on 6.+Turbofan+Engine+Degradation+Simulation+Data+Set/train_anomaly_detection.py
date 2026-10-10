#!/usr/bin/env python3
"""
Anomaly Detection on Combined Aircraft Engine Dataset (Folder 6) using Gaussian Mixture Model (EM).

Workflow:
1. Load train.csv, dev.csv, and test.csv from Processed data/.
2. Select sensor and setting features (dropping zero-variance sensors).
3. Standardize features based on normal operating cycles (RUL > 125 cycles).
4. Train GaussianMixture model with EM on normal training samples.
5. Calibrate log-likelihood anomaly threshold on dev.csv.
6. Evaluate anomaly detection performance on test.csv.
7. Generate and save publication-quality evaluation and trajectory plots.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gmm import GaussianMixture


def prepare_features(df_train, df_dev, df_test, rul_threshold=125):
    """
    Select active features, assign binary anomaly labels based on RUL,
    and standardize features using statistics from normal training cycles.
    """
    feature_cols = [c for c in df_train.columns if c.startswith("s_") or c.startswith("setting_")]

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

    return X_train_normal, X_dev, y_dev, X_test, y_test, active_features, mean, std


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics: Accuracy, Precision, Recall, F1, and Confusion Matrix.
    """
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
    """
    Grid-search optimal threshold on dev set log-likelihood scores to maximize F1.
    """
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


def compute_roc_pr(y_true, anomaly_scores):
    """
    Compute ROC curve (FPR, TPR, AUC) and Precision-Recall curve (Recall, Precision, AP)
    using vectorized NumPy. Anomaly score: higher value indicates greater abnormality.
    """
    desc_idx = np.argsort(anomaly_scores)[::-1]
    y_score = anomaly_scores[desc_idx]
    y_true_sorted = y_true[desc_idx]

    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    roc_auc = float(np.trapezoid(tpr, fpr))

    recalls = np.r_[0.0, tps / n_pos]
    precisions = np.r_[1.0, tps / (tps + fps)]
    # Average Precision (AP)
    ap = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))

    return fpr, tpr, roc_auc, recalls, precisions, ap


def plot_training_and_evaluation(gmm, X_test, y_test, log_p_test, threshold, test_metrics, df_test, output_path):
    """
    Create a 6-panel comprehensive diagnostic dashboard:
    1. EM Convergence Curve
    2. Test Log-Likelihood Distributions (Normal vs Anomaly) + Decision Threshold
    3. Confusion Matrix with Counts & Percentages
    4. ROC Curve with AUC & Calibrated Operating Point
    5. Precision-Recall Curve with AP & Calibrated Operating Point
    6. Log-Likelihood vs Remaining Useful Life (RUL) with Binned Median Trend
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#ffffff")
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    # 1. EM Convergence
    ax1 = axes[0, 0]
    iters = np.arange(len(gmm.log_likelihood_history_))
    ax1.plot(iters, gmm.log_likelihood_history_, marker="o", markersize=4, color="#1e40af", linewidth=2.0)
    ax1.set_title("Gaussian Mixture EM Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("EM Iteration", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Mean Log-Likelihood per Sample", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    final_ll = gmm.log_likelihood_history_[-1]
    ax1.annotate(
        f"Converged at iter {gmm.n_iter_}\nLL = {final_ll:.2f}",
        xy=(gmm.n_iter_, final_ll),
        xytext=(max(1, gmm.n_iter_ - 4), final_ll - (abs(final_ll)*0.02 + 0.5)),
        arrowprops=dict(facecolor="#1e40af", shrink=0.08, width=1.2, headwidth=5),
        fontsize=9,
        fontweight="bold",
        color="#1e40af",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#dbeafe", edgecolor="#93c5fd", alpha=0.9)
    )

    # 2. Score Distributions
    ax2 = axes[0, 1]
    normal_scores = log_p_test[y_test == 0]
    anomaly_scores_ll = log_p_test[y_test == 1]
    q_low = np.percentile(log_p_test, 1)
    q_high = np.percentile(log_p_test, 99)
    bins = np.linspace(q_low, q_high, 50)

    ax2.hist(normal_scores, bins=bins, density=True, alpha=0.55, color="#2563eb", label="Normal (RUL > 125)", edgecolor="#1d4ed8")
    ax2.hist(anomaly_scores_ll, bins=bins, density=True, alpha=0.55, color="#dc2626", label="Degrading (RUL <= 125)", edgecolor="#b91c1c")
    ax2.axvline(threshold, color="#7c3aed", linestyle="--", linewidth=2.2, label=f"Threshold ($\\tau = {threshold:.2f}$)")
    ax2.set_title("Test Log-Likelihood Distributions", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Log-Likelihood $\\log p(x)$", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Confusion Matrix Heatmap
    ax3 = axes[0, 2]
    cm = np.array([
        [test_metrics["tn"], test_metrics["fp"]],
        [test_metrics["fn"], test_metrics["tp"]]
    ])
    im = ax3.imshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    classes = ["Normal\n(RUL > 125)", "Anomaly\n(RUL <= 125)"]
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Pred Normal", "Pred Anomaly"], fontsize=10, fontweight="bold")
    ax3.set_yticklabels(classes, fontsize=10, fontweight="bold")
    ax3.set_title("Test Set Confusion Matrix", fontsize=12, fontweight="bold")

    total_samples = len(y_test)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / total_samples * 100
            row_pct = val / np.sum(cm[i, :]) * 100
            color = "white" if val > cm.max() * 0.55 else "black"
            text_str = f"{val:,}\n({pct:.1f}% total)\n({row_pct:.1f}% class)"
            ax3.text(j, i, text_str, ha="center", va="center", color=color, fontsize=9.5, fontweight="bold")

    stats_str = f"Acc: {test_metrics['accuracy']:.1%} | Prec: {test_metrics['precision']:.1%} | Rec: {test_metrics['recall']:.1%} | F1: {test_metrics['f1']:.1%}"
    ax3.set_xlabel(stats_str, fontsize=10, fontweight="bold", color="#1f2937", labelpad=8)

    # Compute ROC and PR curves (anomaly score = -log p(x))
    anomaly_scores_eval = -log_p_test
    fpr, tpr, roc_auc, recalls, precisions, ap = compute_roc_pr(y_test, anomaly_scores_eval)

    # Operating point on test set
    operating_fpr = test_metrics["fp"] / (test_metrics["fp"] + test_metrics["tn"])
    operating_tpr = test_metrics["tp"] / (test_metrics["tp"] + test_metrics["fn"])

    # 4. ROC Curve
    ax4 = axes[1, 0]
    ax4.plot(fpr, tpr, color="#2563eb", linewidth=2.4, label=f"GMM Model (AUC = {roc_auc:.4f})")
    ax4.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1.5, label="Random Chance (AUC = 0.50)")
    ax4.scatter([operating_fpr], [operating_tpr], color="#dc2626", s=70, zorder=5, label=f"Operating Point (F1={test_metrics['f1']:.2f})")
    ax4.annotate(
        f"FPR={operating_fpr:.2f}, TPR={operating_tpr:.2f}",
        xy=(operating_fpr, operating_tpr),
        xytext=(operating_fpr + 0.08, operating_tpr - 0.12),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.2, headwidth=5),
        fontsize=9,
        fontweight="bold",
        color="#dc2626",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.9)
    )
    ax4.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("False Positive Rate (FPR)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=10, fontweight="bold")
    ax4.set_xlim([-0.02, 1.02])
    ax4.set_ylim([-0.02, 1.02])
    ax4.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # 5. Precision-Recall Curve
    ax5 = axes[1, 1]
    baseline_prec = np.sum(y_test == 1) / len(y_test)
    ax5.plot(recalls, precisions, color="#059669", linewidth=2.4, label=f"PR Curve (AP = {ap:.4f})")
    ax5.axhline(baseline_prec, color="#9ca3af", linestyle="--", linewidth=1.5, label=f"No-skill Baseline ({baseline_prec:.2f})")
    ax5.scatter([test_metrics["recall"]], [test_metrics["precision"]], color="#dc2626", s=70, zorder=5, label="Operating Point")
    ax5.annotate(
        f"Prec={test_metrics['precision']:.2f}, Rec={test_metrics['recall']:.2f}",
        xy=(test_metrics["recall"], test_metrics["precision"]),
        xytext=(test_metrics["recall"] - 0.35, test_metrics["precision"] - 0.15),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.2, headwidth=5),
        fontsize=9,
        fontweight="bold",
        color="#dc2626",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.9)
    )
    ax5.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Recall", fontsize=10, fontweight="bold")
    ax5.set_ylabel("Precision", fontsize=10, fontweight="bold")
    ax5.set_xlim([-0.02, 1.02])
    ax5.set_ylim([-0.02, 1.02])
    ax5.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # 6. Log-Likelihood vs Remaining Useful Life (RUL)
    ax6 = axes[1, 2]
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(df_test), size=min(4000, len(df_test)), replace=False)
    rul_sub = df_test["RUL"].values[sample_idx]
    logp_sub = log_p_test[sample_idx]

    ax6.scatter(rul_sub, logp_sub, alpha=0.18, s=9, color="#3b82f6", label="Test Samples (subsampled)")

    # Binned median curve
    bins_rul = np.linspace(0, 320, 25)
    bin_centers = 0.5 * (bins_rul[:-1] + bins_rul[1:])
    binned_medians = []
    for i in range(len(bins_rul) - 1):
        mask = (df_test["RUL"].values >= bins_rul[i]) & (df_test["RUL"].values < bins_rul[i+1])
        if np.sum(mask) > 10:
            binned_medians.append(np.median(log_p_test[mask]))
        else:
            binned_medians.append(np.nan)

    ax6.plot(bin_centers, binned_medians, color="#dc2626", linewidth=2.8, label="Median $\\log p(x)$ Trend")
    ax6.axhline(threshold, color="#7c3aed", linestyle="--", linewidth=2.0, label=f"Anomaly Threshold ($\\tau = {threshold:.2f}$)")
    ax6.axvline(125, color="#d97706", linestyle=":", linewidth=2.0, label="Degradation Boundary ($RUL = 125$)")
    ax6.axvspan(0, 125, color="#ef4444", alpha=0.08, label="Degradation Regime")
    ax6.set_title("Health Score vs. Remaining Useful Life", fontsize=12, fontweight="bold")
    ax6.set_xlabel("True Remaining Useful Life (RUL in cycles)", fontsize=10, fontweight="bold")
    ax6.set_ylabel("Log-Likelihood $\\log p(x)$", fontsize=10, fontweight="bold")
    ax6.set_xlim([0, 320])
    ax6.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("GMM Anomaly Detection - Evaluation Dashboard (C-MAPSS Folder 6)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Evaluation dashboard saved to: {output_path}")


def plot_engine_trajectories(gmm, features, mean, std, threshold, raw_dir, output_path):
    """
    Plot operational lifecycle degradation trajectories for sample engines (FD001 Unit 1 & Unit 2).
    Shows how log-likelihood health index evolves from healthy operation through degradation to failure,
    demonstrating advance warning lead time.
    """
    train_fd001_file = raw_dir / "train_FD001.txt"
    if not train_fd001_file.exists():
        print(f"Warning: Raw file {train_fd001_file} not found. Skipping engine trajectory plot.")
        return

    cols = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]
    df_raw = pd.read_csv(train_fd001_file, sep=r"\s+", header=None, names=cols)

    units_to_plot = [1, 2]
    fig, axes = plt.subplots(len(units_to_plot), 1, figsize=(14, 9), sharey=True)
    fig.patch.set_facecolor("#ffffff")
    plt.subplots_adjust(hspace=0.35)

    for ax, unit_id in zip(axes, units_to_plot):
        unit_df = df_raw[df_raw["unit_nr"] == unit_id].sort_values("time_cycles").copy()
        max_cycle = unit_df["time_cycles"].max()
        unit_df["RUL"] = max_cycle - unit_df["time_cycles"]

        X_u = (unit_df[features].values - mean) / std
        scores = gmm.score_samples(X_u)
        unit_df["log_p"] = scores
        unit_df["rolling_log_p"] = unit_df["log_p"].rolling(5, min_periods=1).mean()

        cycles = unit_df["time_cycles"].values
        degradation_cycle = max_cycle - 125  # ground truth RUL <= 125 onset

        # Plot instantaneous and smoothed health indices
        ax.plot(cycles, unit_df["log_p"], color="#93c5fd", alpha=0.55, linewidth=1.1, label="Instantaneous $\\log p(x)$")
        ax.plot(cycles, unit_df["rolling_log_p"], color="#1d4ed8", linewidth=2.3, label="5-Cycle Rolling Health Index")
        ax.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Anomaly Threshold ($\\tau = {threshold:.2f}$)")

        if degradation_cycle > 0:
            ax.axvline(degradation_cycle, color="#d97706", linestyle=":", linewidth=1.8, label=f"Degradation Onset (Cycle {degradation_cycle}, RUL $\\leq$ 125)")
            ax.axvspan(0, degradation_cycle, color="#10b981", alpha=0.08, label="Healthy Regime")
            ax.axvspan(degradation_cycle, max_cycle, color="#ef4444", alpha=0.08, label="Degradation Regime")

        # Find first sustained alarm
        anom_cycles = unit_df[unit_df["rolling_log_p"] < threshold]["time_cycles"].values
        if len(anom_cycles) > 0:
            first_alarm = anom_cycles[0]
            lead_time = max_cycle - first_alarm
            ax.plot(first_alarm, threshold, marker="v", markersize=10, color="#b91c1c")
            ax.annotate(
                f"Sustained Alert at Cycle {first_alarm}\n({lead_time} cycles advance warning)",
                xy=(first_alarm, threshold),
                xytext=(first_alarm - 42 if first_alarm > 50 else first_alarm + 10, threshold - 8),
                arrowprops=dict(facecolor="#b91c1c", shrink=0.08, width=1.4, headwidth=5),
                fontsize=9.5,
                fontweight="bold",
                color="#b91c1c",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.9)
            )

        ax.set_title(f"Engine Unit #{unit_id} (FD001): Operational Health Trajectory (Total Lifespan: {max_cycle} cycles)", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Operational Flight Cycles", fontsize=10, fontweight="bold")
        ax.set_ylabel("Log-Likelihood $\\log p(x)$", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    plt.suptitle("C-MAPSS Turbofan Engine Degradation & Condition Monitoring Over Lifespan", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Engine trajectories plot saved to: {output_path}")


def run_anomaly_detection(k=6, max_iter=50, reg_cov=1e-3, seed=42, plot=True, plot_dir=None):
    """
    Train and evaluate GMM anomaly detector on Folder 6 dataset,
    and optionally generate diagnostic and degradation trajectory plots.
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "Processed data"
    raw_dir = base_dir.parent / "data" / "6.+Turbofan+Engine+Degradation+Simulation+Data+Set" / "6. Turbofan Engine Degradation Simulation Data Set" / "CMAPSSData"

    if plot_dir is None:
        plot_dir = base_dir / "plots"
    else:
        plot_dir = Path(plot_dir)

    print("===============================================================")
    print("  GMM Anomaly Detection on Combined Folder 6 Dataset")
    print("===============================================================")
    print(f"Loading data from {data_dir}...")

    df_train = pd.read_csv(data_dir / "train.csv")
    df_dev = pd.read_csv(data_dir / "dev.csv")
    df_test = pd.read_csv(data_dir / "test.csv")

    X_train_normal, X_dev, y_dev, X_test, y_test, features, mean, std = prepare_features(df_train, df_dev, df_test)

    # Subsample normal training set for fast GMM EM convergence if large
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

    saved_plots = []
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)
        eval_plot_path = plot_dir / "gmm_training_and_evaluation.png"
        traj_plot_path = plot_dir / "engine_degradation_trajectories.png"

        print(f"\nGenerating visual plots in {plot_dir}...")
        plot_training_and_evaluation(gmm, X_test, y_test, log_p_test, threshold, test_metrics, df_test, eval_plot_path)
        plot_engine_trajectories(gmm, features, mean, std, threshold, raw_dir, traj_plot_path)

        saved_plots.extend([eval_plot_path, traj_plot_path])
        print(f"Visual plots generated successfully!")

    return gmm, test_metrics, saved_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMM Anomaly Detection on Folder 6")
    parser.add_argument("--k", type=int, default=6, help="Number of Gaussian components")
    parser.add_argument("--max_iter", type=int, default=50, help="Max EM iterations")
    parser.add_argument("--reg_cov", type=float, default=1e-3, help="Covariance regularization")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting visuals after training")
    parser.add_argument("--plot-dir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()

    run_anomaly_detection(
        k=args.k,
        max_iter=args.max_iter,
        reg_cov=args.reg_cov,
        plot=not args.no_plot,
        plot_dir=args.plot_dir
    )
