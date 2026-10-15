#!/usr/bin/env python3
"""
Train and Evaluate Gaussian Mixture Model (GMM) for Turbofan Engine Anomaly Detection
on NASA N-CMAPSS (Dataset 17).

Workflow:
1. Load preprocessed tensors from Data/ (or run preprocess.py if arrays are missing).
2. Fit GMM (k=6) via Expectation-Maximization (EM) on nominal/healthy training cycles.
3. Calibrate log-likelihood anomaly threshold tau on Dev set.
4. Evaluate on unseen Test set (Units 11, 14, 15: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
5. Save trained model bundle to Output/gmm_model.pkl.
6. Generate publication-quality visual diagnostics:
   - Evaluation Dashboard: EM convergence, score distributions, confusion matrix,
     ROC curve, PR curve, RUL vs score.
   - Lifecycle Degradation Trajectories: Individual engine health tracking for test units
     demonstrating advance warning lead time.
"""

from pathlib import Path
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

from gmm import GaussianMixture


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard binary classification metrics and confusion matrix counts."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calibrate_threshold(
    log_probs_dev: np.ndarray,
    y_dev: np.ndarray,
    criterion: str = "balanced_f1",
    seed: int = 42
) -> tuple:
    """
    Calibrate decision threshold tau on Dev set log-likelihoods.
    Criteria:
      - 'balanced_f1' (default): Evaluates candidate thresholds on a balanced subsample of normal and anomaly cycles.
      - 'youden': Maximizes Youden's J statistic (TPR - FPR).
      - 'f1': Maximizes standard F1 on the dev set.
    """
    rng = np.random.default_rng(seed)

    if criterion == "balanced_f1":
        norm_idx = np.where(y_dev == 0)[0]
        deg_idx = np.where(y_dev == 1)[0]
        n_eval = min(len(norm_idx), len(deg_idx), 25000)
        chosen_norm = rng.choice(norm_idx, size=n_eval, replace=False)
        chosen_deg = rng.choice(deg_idx, size=n_eval, replace=False)
        eval_idx = np.concatenate([chosen_norm, chosen_deg])
        s_eval = log_probs_dev[eval_idx]
        y_eval = y_dev[eval_idx]
    else:
        s_eval = log_probs_dev
        y_eval = y_dev

    candidate_percentiles = np.linspace(1, 99, 200)
    candidate_thresholds = np.percentile(s_eval, candidate_percentiles)

    best_thresh = candidate_thresholds[0]
    best_score = -1.0
    best_metrics = None

    if criterion == "youden":
        fpr, tpr, threshs = roc_curve(y_eval, -s_eval)
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        best_thresh = float(-threshs[best_j_idx])
        y_pred = (s_eval < best_thresh).astype(int)
        best_metrics = compute_metrics(y_eval, y_pred)
    else:
        for thresh in candidate_thresholds:
            y_pred = (s_eval < thresh).astype(int)
            metrics = compute_metrics(y_eval, y_pred)
            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_thresh = thresh
                best_metrics = metrics

    return float(best_thresh), best_metrics


def score_in_chunks(gmm: GaussianMixture, X: np.ndarray, chunk_size: int = 100000) -> np.ndarray:
    """Score large arrays in chunks to optimize memory usage."""
    n = len(X)
    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        scores[start:end] = gmm.score_samples(X[start:end])
    return scores


def plot_training_and_evaluation(
    gmm: GaussianMixture,
    y_test: np.ndarray,
    log_p_test: np.ndarray,
    threshold: float,
    test_metrics: dict,
    A_test: np.ndarray,
    Y_test: np.ndarray,
    output_path: Path
):
    """
    Generate 6-panel comprehensive diagnostic evaluation dashboard for Dataset 17.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#ffffff")
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    # 1. EM Convergence Curve
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
        xytext=(max(1, gmm.n_iter_ - 8), final_ll - (abs(final_ll) * 0.05 + 1.0)),
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

    ax2.hist(normal_scores, bins=bins, density=True, alpha=0.55, color="#2563eb", label="Normal ($h_s = 1$)", edgecolor="#1d4ed8")
    ax2.hist(anomaly_scores_ll, bins=bins, density=True, alpha=0.55, color="#dc2626", label="Degraded ($h_s = 0$)", edgecolor="#b91c1c")
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

    classes = ["Normal\n($h_s = 1$)", "Anomaly\n($h_s = 0$)"]
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

    # Compute ROC and PR curves
    anomaly_scores_eval = -log_p_test
    # Subsample for curve computation if test set is very large
    if len(anomaly_scores_eval) > 100000:
        rng = np.random.default_rng(42)
        curve_idx = rng.choice(len(anomaly_scores_eval), size=100000, replace=False)
        y_test_curve = y_test[curve_idx]
        scores_curve = anomaly_scores_eval[curve_idx]
    else:
        y_test_curve = y_test
        scores_curve = anomaly_scores_eval

    fpr, tpr, _ = roc_curve(y_test_curve, scores_curve)
    roc_auc = auc(fpr, tpr)
    precisions, recalls, _ = precision_recall_curve(y_test_curve, scores_curve)
    ap = average_precision_score(y_test_curve, scores_curve)

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
    ax5.axhline(baseline_prec, color="#9ca3af", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline_prec:.2f})")
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
    sample_size = min(5000, len(y_test))
    sample_idx = rng.choice(len(y_test), size=sample_size, replace=False)
    rul_sub = Y_test[sample_idx, 0]
    logp_sub = log_p_test[sample_idx]
    y_sub = y_test[sample_idx]

    ax6.scatter(rul_sub[y_sub == 0], logp_sub[y_sub == 0], alpha=0.25, s=10, color="#2563eb", label="Nominal ($h_s = 1$)")
    ax6.scatter(rul_sub[y_sub == 1], logp_sub[y_sub == 1], alpha=0.25, s=10, color="#dc2626", label="Degraded ($h_s = 0$)")

    # Binned median curve
    max_rul = int(np.percentile(Y_test[:, 0], 99.5))
    bins_rul = np.linspace(0, max_rul, 25)
    bin_centers = 0.5 * (bins_rul[:-1] + bins_rul[1:])
    binned_medians = []
    for i in range(len(bins_rul) - 1):
        mask = (Y_test[:, 0] >= bins_rul[i]) & (Y_test[:, 0] < bins_rul[i + 1])
        if np.sum(mask) > 10:
            binned_medians.append(np.median(log_p_test[mask]))
        else:
            binned_medians.append(np.nan)

    ax6.plot(bin_centers, binned_medians, color="#111827", linewidth=2.8, label="Median $\\log p(x)$ Trend")
    ax6.axhline(threshold, color="#7c3aed", linestyle="--", linewidth=2.0, label=f"Threshold ($\\tau = {threshold:.2f}$)")
    ax6.set_title("Health Score vs. Remaining Useful Life", fontsize=12, fontweight="bold")
    ax6.set_xlabel("True Remaining Useful Life (RUL in flight cycles)", fontsize=10, fontweight="bold")
    ax6.set_ylabel("Log-Likelihood $\\log p(x)$", fontsize=10, fontweight="bold")
    ax6.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("GMM Anomaly Detection - Evaluation Dashboard (NASA N-CMAPSS Dataset 17)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Evaluation dashboard saved to: {output_path}")


def plot_engine_trajectories(
    A_test: np.ndarray,
    log_p_test: np.ndarray,
    threshold: float,
    output_path: Path
):
    """
    Plot operational lifecycle degradation trajectories for test engines (Units 11, 14, 15).
    Tracks instantaneous log-likelihood and rolling smoothed health index against fault onset (ts)
    and end-of-life (tEOL) to evaluate advance warning lead time.
    """
    units = np.unique(A_test[:, 0]).astype(int)
    fig, axes = plt.subplots(len(units), 1, figsize=(15, 11), sharey=True)
    fig.patch.set_facecolor("#ffffff")
    plt.subplots_adjust(hspace=0.38)

    print("\n--- Engine Lifecycle Diagnostics & Early Warning Lead Times ---")
    for ax, unit_id in zip(axes, units):
        u_mask = A_test[:, 0] == unit_id
        u_cycles = A_test[u_mask, 1]
        u_hs = A_test[u_mask, 3]
        u_scores = log_p_test[u_mask]

        unique_cycles = np.sort(np.unique(u_cycles))
        max_cycle = int(unique_cycles.max())

        # Ground truth transition cycle (ts) where hs first becomes 0
        deg_mask = u_hs == 0
        ts_candidates = u_cycles[deg_mask]
        ts = int(ts_candidates.min()) if len(ts_candidates) > 0 else max_cycle

        # Compute cycle medians
        cycle_medians = np.array([np.median(u_scores[u_cycles == c]) for c in unique_cycles])

        # Rolling smoothed trend (window=3)
        rolling_smooth = np.convolve(cycle_medians, np.ones(3)/3, mode='same')
        # Correct boundary effects
        rolling_smooth[0] = cycle_medians[0]
        rolling_smooth[-1] = cycle_medians[-1]

        # Subsample raw points for background scatter
        rng = np.random.default_rng(42)
        n_raw = min(6000, len(u_scores))
        raw_sub = rng.choice(len(u_scores), size=n_raw, replace=False)

        ax.scatter(u_cycles[raw_sub], u_scores[raw_sub], alpha=0.15, s=6, color="#93c5fd", label="Instantaneous $\\log p(x)$")
        ax.plot(unique_cycles, cycle_medians, color="#60a5fa", linewidth=1.2, linestyle="--", label="Cycle Median $\\log p(x)$")
        ax.plot(unique_cycles, rolling_smooth, color="#1e40af", linewidth=2.4, label="Rolling Health Index (Window=3)")
        ax.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Anomaly Threshold ($\\tau = {threshold:.2f}$)")

        # Mark ground-truth degradation onset
        ax.axvline(ts, color="#d97706", linestyle=":", linewidth=2.0, label=f"Fault Onset ($t_s = {ts}$)")
        ax.axvspan(0, ts, color="#10b981", alpha=0.08, label="Nominal Regime ($h_s = 1$)")
        ax.axvspan(ts, max_cycle, color="#ef4444", alpha=0.08, label="Degradation Regime ($h_s = 0$)")

        # Determine sustained alarm cycle (rolling score drops below threshold)
        alarm_cycle = None
        for idx, c in enumerate(unique_cycles):
            if c >= ts and rolling_smooth[idx] < threshold:
                alarm_cycle = int(c)
                break
        if alarm_cycle is None:
            # Fallback search across all cycles
            for idx, c in enumerate(unique_cycles):
                if rolling_smooth[idx] < threshold:
                    alarm_cycle = int(c)
                    break

        lead_time = max_cycle - (alarm_cycle if alarm_cycle else max_cycle)
        print(f"Test Unit {unit_id:2d}: Lifespan = {max_cycle} cycles | Fault Onset ts = {ts} | Sustained Alert = Cycle {alarm_cycle} | Lead Time = {lead_time} cycles")

        if alarm_cycle is not None:
            alarm_y = rolling_smooth[unique_cycles == alarm_cycle][0] if np.any(unique_cycles == alarm_cycle) else threshold
            ax.plot(alarm_cycle, alarm_y, marker="o", markersize=9, color="#dc2626", zorder=6)
            ax.annotate(
                f"Sustained Alert: Cycle {alarm_cycle}\nLead Time: {lead_time} Cycles Advance Warning",
                xy=(alarm_cycle, alarm_y),
                xytext=(max(1, alarm_cycle - 20), alarm_y - 25),
                arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5,
                fontweight="bold",
                color="#dc2626",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.9)
            )

        ax.set_title(f"Test Engine Unit #{unit_id} Operational Lifecycle (Lifespan: {max_cycle} cycles | Fault Onset: Cycle {ts})", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Flight Cycle", fontsize=10, fontweight="bold")
        ax.set_ylabel("Log-Likelihood $\\log p(x)$", fontsize=10, fontweight="bold")
        ax.set_ylim([-140, 45])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9, ncol=3)

    plt.suptitle("NASA N-CMAPSS Test Fleet - Operational Degradation Trajectories & Advance Warning Lead Times", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Lifecycle trajectories saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GMM anomaly detection on NASA N-CMAPSS Dataset 17.")
    parser.add_argument("--k", type=int, default=6, help="Number of GMM mixture components (default: 6)")
    parser.add_argument("--max_iter", type=int, default=50, help="Maximum EM iterations (default: 50)")
    parser.add_argument("--reg_cov", type=float, default=1e-3, help="Covariance diagonal regularization (default: 1e-3)")
    parser.add_argument("--subsample", type=int, default=50000, help="Healthy training cycles subsample (default: 50000)")
    parser.add_argument("--criterion", type=str, default="balanced_f1", choices=["balanced_f1", "youden", "f1"], help="Threshold calibration target (default: balanced_f1)")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing preprocessed .npy files (default: Data/)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for model artifact and diagnostic plots (default: Output/)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-plot", action="store_true", help="Disable generating diagnostic plots")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "Data"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  GMM Anomaly Detection Training & Evaluation (Dataset 17)       ")
    print("=================================================================")

    # Check for preprocessed files
    train_healthy_file = data_dir / "X_train_healthy.npy"
    if not train_healthy_file.exists():
        print(f"Preprocessed data not found in {data_dir}. Running preprocess.py...")
        from preprocess import run_preprocessing
        default_data_file = data_dir / "N-CMAPSS_DS02-006.h5"
        run_preprocessing(data_file=default_data_file, output_dir=data_dir, method="joint")

    print(f"Loading preprocessed arrays from: {data_dir}")
    X_train_h = np.load(data_dir / "X_train_healthy.npy")
    X_dev = np.load(data_dir / "X_dev.npy")
    y_dev = np.load(data_dir / "y_dev.npy")
    A_dev = np.load(data_dir / "A_dev.npy")
    Y_dev = np.load(data_dir / "Y_dev.npy")

    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    A_test = np.load(data_dir / "A_test.npy")
    Y_test = np.load(data_dir / "Y_test.npy")

    with open(data_dir / "pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)

    print(f"Active Features ({len(pipeline['features'])}): {pipeline['features']}")
    print(f"Healthy Train size: {X_train_h.shape[0]:,} samples")
    print(f"Dev size:           {X_dev.shape[0]:,} samples | Anomaly: {np.mean(y_dev)*100:.1f}%")
    print(f"Test size:          {X_test.shape[0]:,} samples | Anomaly: {np.mean(y_test)*100:.1f}%")

    # Subsample training data if larger than requested
    if len(X_train_h) > args.subsample:
        rng = np.random.default_rng(args.seed)
        sub_idx = rng.choice(len(X_train_h), size=args.subsample, replace=False)
        X_train_fit = X_train_h[sub_idx]
    else:
        X_train_fit = X_train_h

    # 1. Fit GMM via EM
    print(f"\n[1/4] Fitting Gaussian Mixture Model (k={args.k}, max_iter={args.max_iter}, reg_cov={args.reg_cov})...")
    gmm = GaussianMixture(
        k=args.k,
        max_iter=args.max_iter,
        reg_cov=args.reg_cov,
        random_state=args.seed
    )
    gmm.fit(X_train_fit)
    print(f"EM Finished: Converged = {gmm.converged_} in {gmm.n_iter_} iterations | Final LL = {gmm.lower_bound_:.2f}")

    # 2. Calibrate Threshold on Dev Set
    print(f"\n[2/4] Calibrating Anomaly Decision Threshold (tau) on Dev Set (criterion: '{args.criterion}')...")
    # Subsample dev for calibration if very large
    dev_n = len(X_dev)
    if dev_n > 200000:
        rng = np.random.default_rng(args.seed)
        calib_idx = rng.choice(dev_n, size=100000, replace=False)
        log_p_dev = score_in_chunks(gmm, X_dev[calib_idx])
        y_dev_calib = y_dev[calib_idx]
    else:
        log_p_dev = score_in_chunks(gmm, X_dev)
        y_dev_calib = y_dev

    threshold, dev_metrics = calibrate_threshold(
        log_probs_dev=log_p_dev,
        y_dev=y_dev_calib,
        criterion=args.criterion,
        seed=args.seed
    )
    print(f"Calibrated Threshold tau = {threshold:.4f}")
    print(f"Dev Set Performance at tau: Acc = {dev_metrics['accuracy']:.2%}, Prec = {dev_metrics['precision']:.2%}, Rec = {dev_metrics['recall']:.2%}, F1 = {dev_metrics['f1']:.2%}")

    # 3. Evaluate on Held-Out Test Set
    print(f"\n[3/4] Evaluating on Held-Out Test Set ({len(X_test):,} unseen operational cycles across Units {pipeline['units_test']})...")
    log_p_test = score_in_chunks(gmm, X_test)
    y_test_pred = (log_p_test < threshold).astype(int)
    test_metrics = compute_metrics(y_test, y_test_pred)

    # Subsample test for ROC-AUC & PR-AUC calculation to prevent excessive memory/time
    if len(y_test) > 200000:
        rng = np.random.default_rng(args.seed)
        auc_idx = rng.choice(len(y_test), size=100000, replace=False)
        roc_auc = roc_auc_score(y_test[auc_idx], -log_p_test[auc_idx])
        pr_auc = average_precision_score(y_test[auc_idx], -log_p_test[auc_idx])
    else:
        roc_auc = roc_auc_score(y_test, -log_p_test)
        pr_auc = average_precision_score(y_test, -log_p_test)

    test_metrics["roc_auc"] = roc_auc
    test_metrics["pr_auc"] = pr_auc

    print("=================================================================")
    print("                    TEST BENCHMARK RESULTS                       ")
    print("=================================================================")
    print(f"Log-Likelihood Threshold (tau): {threshold:.4f}")
    print(f"Accuracy:                       {test_metrics['accuracy']:.2%}")
    print(f"Precision:                      {test_metrics['precision']:.2%}")
    print(f"Recall (Sensitivity):           {test_metrics['recall']:.2%}")
    print(f"F1 Score:                       {test_metrics['f1']:.2%}")
    print(f"ROC-AUC:                        {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC (Average Precision):     {test_metrics['pr_auc']:.4f}")
    print("-----------------------------------------------------------------")
    print(f"Confusion Matrix: TP = {test_metrics['tp']:,} | TN = {test_metrics['tn']:,} | FP = {test_metrics['fp']:,} | FN = {test_metrics['fn']:,}")
    print("=================================================================")

    # 4. Save Model Artifact
    model_bundle = {
        "gmm": gmm,
        "threshold": threshold,
        "mean": pipeline["mean"],
        "std": pipeline["std"],
        "features": pipeline["features"],
        "criterion": args.criterion,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "k": args.k,
        "units_test": pipeline["units_test"],
    }
    model_path = output_dir / "gmm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\nTrained model bundle successfully saved to: {model_path}")

    # 5. Diagnostic Plots
    if not args.no_plot:
        print("\n[4/4] Generating Publication Diagnostic Visualizations...")
        eval_plot_path = output_dir / "gmm_training_and_evaluation.png"
        plot_training_and_evaluation(
            gmm=gmm,
            y_test=y_test,
            log_p_test=log_p_test,
            threshold=threshold,
            test_metrics=test_metrics,
            A_test=A_test,
            Y_test=Y_test,
            output_path=eval_plot_path
        )

        traj_plot_path = output_dir / "engine_degradation_trajectories.png"
        plot_engine_trajectories(
            A_test=A_test,
            log_p_test=log_p_test,
            threshold=threshold,
            output_path=traj_plot_path
        )

    print("\nTraining and Evaluation pipeline finished successfully!")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
