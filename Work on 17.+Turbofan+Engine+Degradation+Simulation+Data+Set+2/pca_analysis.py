#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) Feature Diagnostics for NASA N-CMAPSS (Dataset 17).

This module performs dimensionality reduction, scree analysis, biplot vector projection,
and lifecycle trajectory tracking on the 18 joint flight condition and physical sensor channels.

Outputs:
- Work on 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/Output/pca_feature_analysis.png
"""

from pathlib import Path
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA


def plot_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """Plot an covariance confidence ellipse representing distribution spread."""
    if len(x) < 5:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        facecolor=facecolor,
        **kwargs
    )
    ax.add_patch(ellipse)


def run_pca_dataset_17(
    data_dir: Path,
    output_dir: Path,
    n_components: int = 10,
    subsample_fit: int = 100000,
    subsample_plot: int = 5000,
    dpi: int = 300
):
    """Perform PCA and generate the 6-panel diagnostic dashboard for Dataset 17."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NASA N-CMAPSS (Dataset 17) - Principal Component Analysis (PCA)")
    print("=" * 70)

    # 1. Load preprocessed test telemetry and metadata using memory mapping
    X_test = np.load(data_dir / "X_test.npy", mmap_mode="r")
    y_test = np.load(data_dir / "y_test.npy", mmap_mode="r")
    A_test = np.load(data_dir / "A_test.npy")
    Y_test = np.load(data_dir / "Y_test.npy")

    with open(data_dir / "pipeline.pkl", "rb") as f:
        pipe = pickle.load(f)
    features = pipe["features"]

    print(f"Loaded X_test shape: {X_test.shape} (total samples: {len(X_test):,})")
    print(f"Features ({len(features)}): {features}")

    # 2. Fit PCA on representative subsample
    rng = np.random.default_rng(42)
    fit_indices = rng.choice(len(X_test), size=min(subsample_fit, len(X_test)), replace=False)
    X_fit = np.array(X_test[fit_indices])

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_fit)

    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    print("\n--- Explained Variance Decomposition ---")
    for i in range(min(6, n_components)):
        print(f"  PC{i+1}: {var_ratio[i]*100:6.2f}% (Cumulative: {cum_var[i]*100:6.2f}%)")

    # 3. Transform plot sample
    plot_indices = rng.choice(len(X_test), size=min(subsample_plot, len(X_test)), replace=False)
    X_plot = np.array(X_test[plot_indices])
    Z_plot = pca.transform(X_plot)
    y_plot = np.array(y_test[plot_indices])
    A_plot = A_test[plot_indices]

    # 4. Compute Engine Lifecyle Trajectories across test units (Units 11, 14, 15)
    test_units = np.unique(A_test[:, 0]).astype(int)
    unit_trajectories = {}
    for unit_id in test_units:
        u_mask = A_test[:, 0] == unit_id
        u_cycles = A_test[u_mask, 1]
        u_hs = A_test[u_mask, 3]

        # Subsample points for unit transformation
        u_indices = np.where(u_mask)[0]
        n_unit_sub = min(30000, len(u_indices))
        u_sub_idx = rng.choice(u_indices, size=n_unit_sub, replace=False)

        X_u_sub = np.array(X_test[u_sub_idx])
        Z_u_sub = pca.transform(X_u_sub)
        c_u_sub = A_test[u_sub_idx, 1]

        unique_c = np.sort(np.unique(u_cycles))
        # Compute cycle medians in PC space
        pc1_medians = [np.median(Z_u_sub[c_u_sub == c, 0]) if np.any(c_u_sub == c) else np.nan for c in unique_c]
        pc2_medians = [np.median(Z_u_sub[c_u_sub == c, 1]) if np.any(c_u_sub == c) else np.nan for c in unique_c]

        deg_mask = u_hs == 0
        ts_candidates = u_cycles[deg_mask]
        ts = int(ts_candidates.min()) if len(ts_candidates) > 0 else int(unique_c.max())

        unit_trajectories[unit_id] = {
            "cycles": unique_c,
            "pc1_medians": np.array(pc1_medians),
            "pc2_medians": np.array(pc2_medians),
            "ts": ts,
            "max_cycle": int(unique_c.max())
        }

    # 5. Build 6-Panel Visualization
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#ffffff")
    plt.subplots_adjust(hspace=0.34, wspace=0.28)

    # -------------------------------------------------------------
    # Panel 1: Scree Plot & Cumulative Explained Variance
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    pcs = np.arange(1, n_components + 1)
    bars = ax1.bar(pcs, var_ratio * 100, color="#1e40af", alpha=0.85, edgecolor="#172554", label="Individual Variance (%)")
    ax1.set_xlabel("Principal Component", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Explained Variance (%)", fontsize=10, fontweight="bold", color="#1e40af")
    ax1.set_title("N-CMAPSS Scree Plot & Cumulative Variance", fontsize=12, fontweight="bold")
    ax1.set_xticks(pcs)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)

    for i in range(min(3, n_components)):
        height = bars[i].get_height()
        ax1.text(bars[i].get_x() + bars[i].get_width()/2., height + 1.5,
                 f"{height:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color="#1e40af")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(pcs, cum_var * 100, color="#dc2626", marker="o", linewidth=2.2, markersize=5, label="Cumulative Variance (%)")
    ax1_twin.set_ylabel("Cumulative Explained Variance (%)", fontsize=10, fontweight="bold", color="#dc2626")
    ax1_twin.set_ylim(0, 105)
    ax1_twin.axhline(95, color="#059669", linestyle=":", linewidth=1.5, label="95% Threshold")
    ax1_twin.grid(False)

    ax1.annotate(
        f"PC1 + PC2 capture {cum_var[1]*100:.2f}%\nof 18D mission telemetry",
        xy=(2, cum_var[1]*100),
        xytext=(3.5, 60),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.2, headwidth=5),
        fontsize=9, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#fca5a5", alpha=0.95)
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: PCA Biplot (PC1 vs PC2) with Operational & Sensor Vectors
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.scatter(Z_plot[:, 0], Z_plot[:, 1], c="#cbd5e1", alpha=0.45, s=10, edgecolors="none", label="Test Telemetry")

    loadings = pca.components_[:2].T  # shape (18, 2)
    magnitudes = np.linalg.norm(loadings, axis=1)
    top_indices = np.argsort(magnitudes)[::-1][:12]

    scale_factor = 9.0
    scenario_vars = ["alt", "Mach", "TRA", "T2"]
    for idx in top_indices:
        feat_name = features[idx]
        dx = loadings[idx, 0] * scale_factor
        dy = loadings[idx, 1] * scale_factor
        is_scenario = feat_name in scenario_vars
        color = "#7c3aed" if is_scenario else "#2563eb"
        ax2.arrow(0, 0, dx, dy, color=color, alpha=0.85, width=0.08, head_width=0.35, length_includes_head=True)
        ax2.text(dx * 1.14, dy * 1.14, feat_name, color=color, fontsize=8.5, fontweight="bold", ha="center", va="center")

    ax2.axhline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax2.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax2.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax2.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax2.set_title("N-CMAPSS Biplot: Flight Descriptors & Sensors", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2.plot([], [], color="#7c3aed", marker=">", linewidth=1.5, label="Flight Descriptors (W)")
    ax2.plot([], [], color="#2563eb", marker=">", linewidth=1.5, label="Physical Sensors (X_s)")
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 3: 2D Score Plot: Nominal (hs=1) vs Degraded (hs=0)
    # -------------------------------------------------------------
    ax3 = axes[0, 2]
    nom_mask = y_plot == 0  # hs == 1
    anom_mask = y_plot == 1  # hs == 0

    ax3.scatter(Z_plot[nom_mask, 0], Z_plot[nom_mask, 1],
                c="#2563eb", alpha=0.45, s=14, label="Nominal Flight ($h_s = 1.0$)", edgecolors="none")
    ax3.scatter(Z_plot[anom_mask, 0], Z_plot[anom_mask, 1],
                c="#dc2626", alpha=0.45, s=14, label="Degraded Regime ($h_s = 0.0$)", edgecolors="none")

    plot_confidence_ellipse(Z_plot[nom_mask, 0], Z_plot[nom_mask, 1], ax3, n_std=1.5,
                            edgecolor="#1d4ed8", linewidth=2.0, linestyle="-")
    plot_confidence_ellipse(Z_plot[anom_mask, 0], Z_plot[anom_mask, 1], ax3, n_std=1.5,
                            edgecolor="#b91c1c", linewidth=2.0, linestyle="--")

    ax3.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax3.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax3.set_title("Latent Space by Health State Flag ($h_s$)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 4: Flight Envelope Profiles (Flight Class Fc: 1, 2, 3)
    # -------------------------------------------------------------
    ax4 = axes[1, 0]
    fc_colors = {1: "#2563eb", 2: "#059669", 3: "#d97706"}
    fc_labels = {1: "Flight Class 1 (Short)", 2: "Flight Class 2 (Medium)", 3: "Flight Class 3 (Long)"}
    for fc_val in [1, 2, 3]:
        fc_mask = A_plot[:, 2] == fc_val
        if np.any(fc_mask):
            ax4.scatter(Z_plot[fc_mask, 0], Z_plot[fc_mask, 1],
                        c=fc_colors[fc_val], alpha=0.5, s=14, label=fc_labels[fc_val], edgecolors="none")

    ax4.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax4.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax4.set_title("Latent Space by Commercial Flight Class (Fc)", fontsize=12, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 5: Feature Loadings Bar Chart (PC1 & PC2)
    # -------------------------------------------------------------
    ax5 = axes[1, 1]
    y_pos = np.arange(len(features))
    bar_width = 0.38

    ax5.barh(y_pos + bar_width/2, pca.components_[0], height=bar_width, color="#1e40af", alpha=0.85, label="PC1 Loadings (68.7%)")
    ax5.barh(y_pos - bar_width/2, pca.components_[1], height=bar_width, color="#059669", alpha=0.85, label="PC2 Loadings (29.6%)")
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(features, fontsize=8.5)
    ax5.set_xlabel("Eigenvector Loading Weight", fontsize=10, fontweight="bold")
    ax5.set_title("Principal Component Feature Loadings", fontsize=12, fontweight="bold")
    ax5.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax5.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 6: Engine Degradation Lifecycle Trajectories (Units 11, 14, 15)
    # -------------------------------------------------------------
    ax6 = axes[1, 2]
    traj_colors = {11: "#2563eb", 14: "#059669", 15: "#d97706"}
    for unit_id in test_units:
        t_data = unit_trajectories[unit_id]
        c_vals = t_data["cycles"]
        pc1_m = t_data["pc1_medians"]
        pc2_m = t_data["pc2_medians"]
        valid = ~np.isnan(pc1_m) & ~np.isnan(pc2_m)
        c_vals = c_vals[valid]
        pc1_m = pc1_m[valid]
        pc2_m = pc2_m[valid]

        col = traj_colors.get(unit_id, "#334155")
        ax6.plot(pc1_m, pc2_m, color=col, linewidth=1.8, alpha=0.85, label=f"Unit #{unit_id} Lifespan")

        # Mark Healthy Start (Cycle 1)
        ax6.plot(pc1_m[0], pc2_m[0], marker="o", markersize=7, color=col, markeredgecolor="black", markeredgewidth=1.2)
        # Mark Transition cycle ts
        ts_idx = np.where(c_vals >= t_data["ts"])[0]
        if len(ts_idx) > 0:
            first_ts = ts_idx[0]
            ax6.plot(pc1_m[first_ts], pc2_m[first_ts], marker="s", markersize=8, color="#f59e0b",
                     markeredgecolor="#78350f", markeredgewidth=1.2)
        # Mark EOL failure
        ax6.plot(pc1_m[-1], pc2_m[-1], marker="X", markersize=10, color="#dc2626",
                 markeredgecolor="#7f1d1d", markeredgewidth=1.2)

    # Proxy legends
    ax6.plot([], [], marker="o", color="#334155", linestyle="none", label="Cycle 1 Start")
    ax6.plot([], [], marker="s", color="#f59e0b", linestyle="none", label="Fault Onset ($t_s$)")
    ax6.plot([], [], marker="X", color="#dc2626", linestyle="none", label="End-of-Life Failure")

    ax6.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax6.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax6.set_title("Test Fleet Degradation Trajectories (Units 11, 14, 15)", fontsize=12, fontweight="bold")
    ax6.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax6.grid(True, linestyle="--", alpha=0.5)

    output_path = output_dir / "pca_feature_analysis.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] N-CMAPSS PCA diagnostic dashboard generated at: {output_path}")

    return {
        "var_ratio": var_ratio,
        "cum_var": cum_var,
        "top_pc1_features": sorted(zip(features, pca.components_[0]), key=lambda x: abs(x[1]), reverse=True)[:5],
        "top_pc2_features": sorted(zip(features, pca.components_[1]), key=lambda x: abs(x[1]), reverse=True)[:5]
    }


def main():
    parser = argparse.ArgumentParser(description="PCA Feature Diagnostics for NASA N-CMAPSS (Dataset 17)")
    parser.add_argument("--data-dir", type=str, default="Work on 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/Data")
    parser.add_argument("--output-dir", type=str, default="Work on 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2/Output")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    run_pca_dataset_17(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
