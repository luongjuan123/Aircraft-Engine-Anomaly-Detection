#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) Feature Diagnostics for NASA C-MAPSS (Dataset 6).

This module performs dimensionality reduction, scree analysis, biplot vector projection,
and lifecycle trajectory tracking on the 24 operational and sensor channels of C-MAPSS.

Outputs:
- Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/Output/pca_feature_analysis.png
"""

from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


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


def run_pca_dataset_6(
    data_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    n_components: int = 10,
    dpi: int = 300
):
    """Perform PCA and generate the 6-panel diagnostic dashboard for Dataset 6."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NASA C-MAPSS (Dataset 6) - Principal Component Analysis (PCA)")
    print("=" * 70)

    # 1. Load preprocessed dev data and feature pipeline
    X_dev = np.load(data_dir / "X_dev.npy")
    y_dev = np.load(data_dir / "y_dev.npy")
    df_dev = pd.read_csv(data_dir / "dev.csv")

    with open(data_dir / "pipeline.pkl", "rb") as f:
        pipe = pickle.load(f)
    features = pipe["features"]
    mean_stats = pipe["mean"]
    std_stats = pipe["std"]

    print(f"Loaded X_dev shape: {X_dev.shape}")
    print(f"Features ({len(features)}): {features}")

    # 2. Fit PCA
    pca = PCA(n_components=n_components, random_state=42)
    Z_dev = pca.fit_transform(X_dev)

    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    print("\n--- Explained Variance Decomposition ---")
    for i in range(min(6, n_components)):
        print(f"  PC{i+1}: {var_ratio[i]*100:6.2f}% (Cumulative: {cum_var[i]*100:6.2f}%)")

    # 3. Cluster flight settings into 6 operational regimes
    setting_cols = ["setting_1", "setting_2", "setting_3"]
    km = KMeans(n_clusters=6, random_state=42, n_init=10)
    regimes = km.fit_predict(df_dev[setting_cols].values)

    # 4. Load raw FD001 data for individual engine lifecycle trajectory
    cols_raw = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]
    train_fd001_file = raw_dir / "train_FD001.txt"
    df_unit1 = None
    if train_fd001_file.exists():
        df_raw_fd001 = pd.read_csv(train_fd001_file, sep=r"\s+", header=None, names=cols_raw)
        df_unit1 = df_raw_fd001[df_raw_fd001["unit_nr"] == 1].sort_values("time_cycles").copy()
        X_u1 = (df_unit1[features].values - mean_stats) / std_stats
        Z_u1 = pca.transform(X_u1)
        df_unit1["PC1"] = Z_u1[:, 0]
        df_unit1["PC2"] = Z_u1[:, 1]
        df_unit1["RUL"] = df_unit1["time_cycles"].max() - df_unit1["time_cycles"]

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
    ax1.set_title("Scree Plot & Cumulative Explained Variance", fontsize=12, fontweight="bold")
    ax1.set_xticks(pcs)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Annotate top 2 bars
    for i in range(min(3, n_components)):
        height = bars[i].get_height()
        ax1.text(bars[i].get_x() + bars[i].get_width()/2., height + 1.5,
                 f"{height:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color="#1e40af")

    # Cumulative variance line on secondary axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(pcs, cum_var * 100, color="#dc2626", marker="o", linewidth=2.2, markersize=5, label="Cumulative Variance (%)")
    ax1_twin.set_ylabel("Cumulative Explained Variance (%)", fontsize=10, fontweight="bold", color="#dc2626")
    ax1_twin.set_ylim(0, 105)
    ax1_twin.axhline(95, color="#059669", linestyle=":", linewidth=1.5, label="95% Threshold")
    ax1_twin.grid(False)

    ax1.annotate(
        f"PC1 + PC2 capture {cum_var[1]*100:.2f}%\nof 24D sensor space",
        xy=(2, cum_var[1]*100),
        xytext=(3.5, 60),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.2, headwidth=5),
        fontsize=9, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#fca5a5", alpha=0.95)
    )

    # Combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: PCA Biplot (PC1 vs PC2) with Sensor Loading Vectors
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    # Scatter background points
    sub_idx = np.random.RandomState(42).choice(len(Z_dev), size=min(4000, len(Z_dev)), replace=False)
    ax2.scatter(Z_dev[sub_idx, 0], Z_dev[sub_idx, 1], c="#cbd5e1", alpha=0.45, s=10, edgecolors="none", label="Dev Telemetry")

    # Plot top feature vectors
    loadings = pca.components_[:2].T  # shape (24, 2)
    # Find features with highest vector magnitude
    magnitudes = np.linalg.norm(loadings, axis=1)
    top_feat_indices = np.argsort(magnitudes)[::-1][:12]  # top 12 vectors for clarity

    scale_factor = 8.5  # scale arrows for visualization
    for idx in top_feat_indices:
        feat_name = features[idx]
        dx = loadings[idx, 0] * scale_factor
        dy = loadings[idx, 1] * scale_factor
        is_setting = feat_name.startswith("setting_")
        color = "#7c3aed" if is_setting else "#2563eb"
        ax2.arrow(0, 0, dx, dy, color=color, alpha=0.85, width=0.08, head_width=0.35, length_includes_head=True)
        # Offset label slightly
        offset_x = dx * 1.12
        offset_y = dy * 1.12
        ax2.text(offset_x, offset_y, feat_name, color=color, fontsize=8.5, fontweight="bold", ha="center", va="center")

    ax2.axhline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax2.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax2.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax2.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax2.set_title("PCA Biplot: Sensor & Setting Loadings", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Dummy proxies for legend
    ax2.plot([], [], color="#2563eb", marker=">", linewidth=1.5, label="Sensor Channels (s_i)")
    ax2.plot([], [], color="#7c3aed", marker=">", linewidth=1.5, label="Flight Settings")
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 3: 2D Score Plot Colored by Health State (Nominal vs Degraded)
    # -------------------------------------------------------------
    ax3 = axes[0, 2]
    norm_mask = y_dev[sub_idx] == 0
    anom_mask = y_dev[sub_idx] == 1

    ax3.scatter(Z_dev[sub_idx][norm_mask, 0], Z_dev[sub_idx][norm_mask, 1],
                c="#2563eb", alpha=0.5, s=14, label="Nominal (RUL > 125)", edgecolors="none")
    ax3.scatter(Z_dev[sub_idx][anom_mask, 0], Z_dev[sub_idx][anom_mask, 1],
                c="#dc2626", alpha=0.5, s=14, label="Degrading (RUL \u2264 125)", edgecolors="none")

    # Add confidence ellipses for dispersion
    plot_confidence_ellipse(Z_dev[y_dev == 0, 0], Z_dev[y_dev == 0, 1], ax3, n_std=1.5,
                            edgecolor="#1d4ed8", linewidth=2.0, linestyle="-")
    plot_confidence_ellipse(Z_dev[y_dev == 1, 0], Z_dev[y_dev == 1, 1], ax3, n_std=1.5,
                            edgecolor="#b91c1c", linewidth=2.0, linestyle="--")

    ax3.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax3.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax3.set_title("Latent Space by Health Degradation State", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 4: 2D Score Plot Colored by 6 Operational Regimes
    # -------------------------------------------------------------
    ax4 = axes[1, 0]
    regime_colors = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2"]
    for r in range(6):
        r_mask = regimes[sub_idx] == r
        ax4.scatter(Z_dev[sub_idx][r_mask, 0], Z_dev[sub_idx][r_mask, 1],
                    c=regime_colors[r], alpha=0.55, s=14, label=f"Regime {r+1}", edgecolors="none")

    ax4.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax4.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax4.set_title("Latent Space by Operational Flight Regime (k=6)", fontsize=12, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 5: Feature Loadings Bar Chart (PC1 & PC2)
    # -------------------------------------------------------------
    ax5 = axes[1, 1]
    y_pos = np.arange(len(features))
    bar_width = 0.38

    ax5.barh(y_pos + bar_width/2, pca.components_[0], height=bar_width, color="#1e40af", alpha=0.85, label="PC1 Loadings (82.3%)")
    ax5.barh(y_pos - bar_width/2, pca.components_[1], height=bar_width, color="#059669", alpha=0.85, label="PC2 Loadings (16.0%)")
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(features, fontsize=8)
    ax5.set_xlabel("Eigenvector Loading Weight", fontsize=10, fontweight="bold")
    ax5.set_title("Principal Component Feature Loadings", fontsize=12, fontweight="bold")
    ax5.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax5.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 6: Engine Lifecycle Trajectory in PC Space (Unit #1)
    # -------------------------------------------------------------
    ax6 = axes[1, 2]
    if df_unit1 is not None:
        scatter = ax6.scatter(
            df_unit1["PC1"], df_unit1["PC2"],
            c=df_unit1["time_cycles"],
            cmap="turbo",
            s=22,
            edgecolors="k",
            linewidths=0.3,
            label="Flight Cycles"
        )
        # Draw smooth trajectory line
        ax6.plot(df_unit1["PC1"], df_unit1["PC2"], color="#334155", alpha=0.45, linewidth=1.0, linestyle=":")

        # Mark Start (Cycle 1) and End-of-Life (Failure)
        ax6.plot(df_unit1["PC1"].iloc[0], df_unit1["PC2"].iloc[0], marker="o", markersize=10,
                 color="#10b981", markeredgecolor="#064e3b", markeredgewidth=1.8, label="Cycle 1 (Healthy)")
        ax6.plot(df_unit1["PC1"].iloc[-1], df_unit1["PC2"].iloc[-1], marker="X", markersize=12,
                 color="#dc2626", markeredgecolor="#7f1d1d", markeredgewidth=1.8, label="EOL Failure")

        cbar = fig.colorbar(scatter, ax=ax6, fraction=0.046, pad=0.04)
        cbar.set_label("Operational Flight Cycle", fontsize=9, fontweight="bold")

        ax6.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
        ax6.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
        ax6.set_title("Engine #1 Lifecycle Degradation Trajectory (FD001)", fontsize=12, fontweight="bold")
        ax6.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        ax6.grid(True, linestyle="--", alpha=0.5)
    else:
        ax6.text(0.5, 0.5, "Raw FD001 trajectory data not available", ha="center", va="center")

    output_path = output_dir / "pca_feature_analysis.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] PCA diagnostic dashboard generated at: {output_path}")

    # Return key metrics
    return {
        "var_ratio": var_ratio,
        "cum_var": cum_var,
        "top_pc1_features": sorted(zip(features, pca.components_[0]), key=lambda x: abs(x[1]), reverse=True)[:5],
        "top_pc2_features": sorted(zip(features, pca.components_[1]), key=lambda x: abs(x[1]), reverse=True)[:5]
    }


def main():
    parser = argparse.ArgumentParser(description="PCA Feature Diagnostics for NASA C-MAPSS (Dataset 6)")
    parser.add_argument("--data-dir", type=str, default="Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/Data")
    parser.add_argument("--raw-dir", type=str, default="data/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData")
    parser.add_argument("--output-dir", type=str, default="Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set/Output")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    run_pca_dataset_6(
        data_dir=Path(args.data_dir),
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
