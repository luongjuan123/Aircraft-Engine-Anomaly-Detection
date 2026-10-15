#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) Feature Diagnostics & Visualizations for
both NASA Turbofan Engine Degradation Datasets:
1. NASA C-MAPSS (Dataset 6 - Folder 6)
2. NASA N-CMAPSS (Dataset 17 - Folder 17)
3. Cross-Dataset Comparative Benchmark Dashboard

Usage:
    python plot_pca_features.py --dataset all
    python plot_pca_features.py --dataset 6
    python plot_pca_features.py --dataset 17
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
    """Plot an empirical covariance confidence ellipse representing distribution spread."""
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


def plot_dataset_6_pca(base_dir: Path, dpi: int = 300):
    """Generate 6-panel publication-grade PCA diagnostic dashboard for NASA C-MAPSS (Dataset 6)."""
    p6_dir = base_dir / "Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set"
    data_dir = p6_dir / "Data"
    output_dir = p6_dir / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = base_dir / "data/6.+Turbofan+Engine+Degradation+Simulation+Data+Set/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData"

    print("\n" + "=" * 75)
    print("Generating PCA Feature Diagnostics: NASA C-MAPSS (Dataset 6)")
    print("=" * 75)

    X_dev = np.load(data_dir / "X_dev.npy")
    y_dev = np.load(data_dir / "y_dev.npy")
    df_dev = pd.read_csv(data_dir / "dev.csv")

    with open(data_dir / "pipeline.pkl", "rb") as f:
        pipe = pickle.load(f)
    features = pipe["features"]
    mean_stats = pipe["mean"]
    std_stats = pipe["std"]

    pca = PCA(n_components=10, random_state=42)
    Z_dev = pca.fit_transform(X_dev)
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    print(f"Loaded {len(X_dev):,} dev cycles with {len(features)} active channels.")
    print(f"Variance Explained: PC1={var_ratio[0]*100:.2f}%, PC2={var_ratio[1]*100:.2f}%, PC3={var_ratio[2]*100:.2f}%")
    print(f"Cumulative Top-2 PCs: {cum_var[1]*100:.2f}% | Top-3 PCs: {cum_var[2]*100:.2f}%")

    # Cluster settings into 6 regimes
    setting_cols = ["setting_1", "setting_2", "setting_3"]
    km = KMeans(n_clusters=6, random_state=42, n_init=10)
    regimes = km.fit_predict(df_dev[setting_cols].values)

    # Lifecycle trajectory for Unit 1 FD001
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

    # Create figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#ffffff")
    plt.subplots_adjust(hspace=0.34, wspace=0.28)

    # 1. Scree Plot
    ax1 = axes[0, 0]
    pcs = np.arange(1, 11)
    bars = ax1.bar(pcs, var_ratio * 100, color="#1e40af", alpha=0.85, edgecolor="#172554", label="Individual Variance (%)")
    ax1.set_xlabel("Principal Component", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Explained Variance (%)", fontsize=10, fontweight="bold", color="#1e40af")
    ax1.set_title("C-MAPSS Scree Plot & Cumulative Variance", fontsize=12, fontweight="bold")
    ax1.set_xticks(pcs)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)

    for i in range(3):
        h = bars[i].get_height()
        ax1.text(bars[i].get_x() + bars[i].get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color="#1e40af")

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

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8.5, framealpha=0.9)

    # 2. Biplot
    ax2 = axes[0, 1]
    sub_idx = np.random.RandomState(42).choice(len(Z_dev), size=min(4000, len(Z_dev)), replace=False)
    ax2.scatter(Z_dev[sub_idx, 0], Z_dev[sub_idx, 1], c="#cbd5e1", alpha=0.45, s=10, edgecolors="none", label="Dev Telemetry")

    loadings = pca.components_[:2].T
    magnitudes = np.linalg.norm(loadings, axis=1)
    top_indices = np.argsort(magnitudes)[::-1][:12]
    scale_factor = 8.5
    for idx in top_indices:
        feat_name = features[idx]
        dx = loadings[idx, 0] * scale_factor
        dy = loadings[idx, 1] * scale_factor
        is_setting = feat_name.startswith("setting_")
        color = "#7c3aed" if is_setting else "#2563eb"
        ax2.arrow(0, 0, dx, dy, color=color, alpha=0.85, width=0.08, head_width=0.35, length_includes_head=True)
        ax2.text(dx * 1.12, dy * 1.12, feat_name, color=color, fontsize=8.5, fontweight="bold", ha="center", va="center")

    ax2.axhline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax2.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax2.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax2.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax2.set_title("C-MAPSS Biplot: Sensor & Setting Loadings", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2.plot([], [], color="#2563eb", marker=">", linewidth=1.5, label="Sensor Channels (s_i)")
    ax2.plot([], [], color="#7c3aed", marker=">", linewidth=1.5, label="Flight Settings")
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # 3. Healthy vs Degraded
    ax3 = axes[0, 2]
    norm_mask = y_dev[sub_idx] == 0
    anom_mask = y_dev[sub_idx] == 1
    ax3.scatter(Z_dev[sub_idx][norm_mask, 0], Z_dev[sub_idx][norm_mask, 1],
                c="#2563eb", alpha=0.5, s=14, label="Nominal (RUL > 125)", edgecolors="none")
    ax3.scatter(Z_dev[sub_idx][anom_mask, 0], Z_dev[sub_idx][anom_mask, 1],
                c="#dc2626", alpha=0.5, s=14, label="Degrading (RUL \u2264 125)", edgecolors="none")

    plot_confidence_ellipse(Z_dev[y_dev == 0, 0], Z_dev[y_dev == 0, 1], ax3, n_std=1.5,
                            edgecolor="#1d4ed8", linewidth=2.0, linestyle="-")
    plot_confidence_ellipse(Z_dev[y_dev == 1, 0], Z_dev[y_dev == 1, 1], ax3, n_std=1.5,
                            edgecolor="#b91c1c", linewidth=2.0, linestyle="--")

    ax3.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax3.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax3.set_title("Latent Space by Health Degradation State", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Flight Regimes
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

    # 5. Loadings
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

    # 6. Trajectory
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
        ax6.plot(df_unit1["PC1"], df_unit1["PC2"], color="#334155", alpha=0.45, linewidth=1.0, linestyle=":")
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

    out_file = output_dir / "pca_feature_analysis.png"
    plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Done] C-MAPSS PCA Dashboard saved to: {out_file}")


def plot_dataset_17_pca(base_dir: Path, dpi: int = 300):
    """Generate 6-panel publication-grade PCA diagnostic dashboard for NASA N-CMAPSS (Dataset 17)."""
    p17_dir = base_dir / "Work on 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2"
    data_dir = p17_dir / "Data"
    output_dir = p17_dir / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("Generating PCA Feature Diagnostics: NASA N-CMAPSS (Dataset 17)")
    print("=" * 75)

    X_test = np.load(data_dir / "X_test.npy", mmap_mode="r")
    y_test = np.load(data_dir / "y_test.npy", mmap_mode="r")
    A_test = np.load(data_dir / "A_test.npy")

    with open(data_dir / "pipeline.pkl", "rb") as f:
        pipe = pickle.load(f)
    features = pipe["features"]

    rng = np.random.default_rng(42)
    fit_idx = rng.choice(len(X_test), size=min(100000, len(X_test)), replace=False)
    X_fit = np.array(X_test[fit_idx])

    pca = PCA(n_components=10, random_state=42)
    pca.fit(X_fit)
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    print(f"Loaded {len(X_test):,} test telemetry cycles with {len(features)} joint channels.")
    print(f"Variance Explained: PC1={var_ratio[0]*100:.2f}%, PC2={var_ratio[1]*100:.2f}%, PC3={var_ratio[2]*100:.2f}%")
    print(f"Cumulative Top-2 PCs: {cum_var[1]*100:.2f}% | Top-3 PCs: {cum_var[2]*100:.2f}%")

    # Sample points for fast plotting
    plot_idx = rng.choice(len(X_test), size=5000, replace=False)
    X_plot = np.array(X_test[plot_idx])
    Z_plot = pca.transform(X_plot)
    y_plot = np.array(y_test[plot_idx])
    A_plot = A_test[plot_idx]

    # Lifecycle trajectories
    test_units = np.unique(A_test[:, 0]).astype(int)
    unit_trajectories = {}
    for unit_id in test_units:
        u_mask = A_test[:, 0] == unit_id
        u_cycles = A_test[u_mask, 1]
        u_hs = A_test[u_mask, 3]

        u_indices = np.where(u_mask)[0]
        n_unit_sub = min(30000, len(u_indices))
        u_sub_idx = rng.choice(u_indices, size=n_unit_sub, replace=False)

        X_u_sub = np.array(X_test[u_sub_idx])
        Z_u_sub = pca.transform(X_u_sub)
        c_u_sub = A_test[u_sub_idx, 1]

        unique_c = np.sort(np.unique(u_cycles))
        pc1_m = [np.median(Z_u_sub[c_u_sub == c, 0]) if np.any(c_u_sub == c) else np.nan for c in unique_c]
        pc2_m = [np.median(Z_u_sub[c_u_sub == c, 1]) if np.any(c_u_sub == c) else np.nan for c in unique_c]

        deg_mask = u_hs == 0
        ts_candidates = u_cycles[deg_mask]
        ts = int(ts_candidates.min()) if len(ts_candidates) > 0 else int(unique_c.max())

        unit_trajectories[unit_id] = {
            "cycles": unique_c,
            "pc1_medians": np.array(pc1_m),
            "pc2_medians": np.array(pc2_m),
            "ts": ts
        }

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#ffffff")
    plt.subplots_adjust(hspace=0.34, wspace=0.28)

    # 1. Scree Plot
    ax1 = axes[0, 0]
    pcs = np.arange(1, 11)
    bars = ax1.bar(pcs, var_ratio * 100, color="#1e40af", alpha=0.85, edgecolor="#172554", label="Individual Variance (%)")
    ax1.set_xlabel("Principal Component", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Explained Variance (%)", fontsize=10, fontweight="bold", color="#1e40af")
    ax1.set_title("N-CMAPSS Scree Plot & Cumulative Variance", fontsize=12, fontweight="bold")
    ax1.set_xticks(pcs)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)

    for i in range(3):
        h = bars[i].get_height()
        ax1.text(bars[i].get_x() + bars[i].get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color="#1e40af")

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

    # 2. Biplot
    ax2 = axes[0, 1]
    ax2.scatter(Z_plot[:, 0], Z_plot[:, 1], c="#cbd5e1", alpha=0.45, s=10, edgecolors="none", label="Test Telemetry")

    loadings = pca.components_[:2].T
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

    # 3. Healthy vs Degraded
    ax3 = axes[0, 2]
    nom_mask = y_plot == 0
    anom_mask = y_plot == 1
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

    # 4. Commercial Flight Class
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

    # 5. Loadings
    ax5 = axes[1, 1]
    y_pos = np.arange(len(features))
    bar_width = 0.38
    ax5.barh(y_pos + bar_width/2, pca.components_[0], height=bar_width, color="#1e40af", alpha=0.85, label="PC1 Loadings (68.4%)")
    ax5.barh(y_pos - bar_width/2, pca.components_[1], height=bar_width, color="#059669", alpha=0.85, label="PC2 Loadings (30.0%)")
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(features, fontsize=8.5)
    ax5.set_xlabel("Eigenvector Loading Weight", fontsize=10, fontweight="bold")
    ax5.set_title("Principal Component Feature Loadings", fontsize=12, fontweight="bold")
    ax5.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax5.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # 6. Trajectories
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
        ax6.plot(pc1_m[0], pc2_m[0], marker="o", markersize=7, color=col, markeredgecolor="black", markeredgewidth=1.2)
        ts_idx = np.where(c_vals >= t_data["ts"])[0]
        if len(ts_idx) > 0:
            first_ts = ts_idx[0]
            ax6.plot(pc1_m[first_ts], pc2_m[first_ts], marker="s", markersize=8, color="#f59e0b",
                     markeredgecolor="#78350f", markeredgewidth=1.2)
        ax6.plot(pc1_m[-1], pc2_m[-1], marker="X", markersize=10, color="#dc2626",
                 markeredgecolor="#7f1d1d", markeredgewidth=1.2)

    ax6.plot([], [], marker="o", color="#334155", linestyle="none", label="Cycle 1 Start")
    ax6.plot([], [], marker="s", color="#f59e0b", linestyle="none", label="Fault Onset ($t_s$)")
    ax6.plot([], [], marker="X", color="#dc2626", linestyle="none", label="End-of-Life Failure")

    ax6.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax6.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax6.set_title("Test Fleet Degradation Trajectories (Units 11, 14, 15)", fontsize=12, fontweight="bold")
    ax6.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax6.grid(True, linestyle="--", alpha=0.5)

    out_file = output_dir / "pca_feature_analysis.png"
    plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Done] N-CMAPSS PCA Dashboard saved to: {out_file}")


def plot_cross_dataset_comparison(base_dir: Path, dpi: int = 300):
    """Generate 4-panel Cross-Dataset Comparative PCA Dashboard."""
    p6_dir = base_dir / "Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set"
    p17_dir = base_dir / "Work on 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2"

    print("\n" + "=" * 75)
    print("Generating Cross-Dataset Comparative PCA Dashboard")
    print("=" * 75)

    # Dataset 6
    X_dev_6 = np.load(p6_dir / "Data/X_dev.npy")
    y_dev_6 = np.load(p6_dir / "Data/y_dev.npy")
    with open(p6_dir / "Data/pipeline.pkl", "rb") as f:
        f6 = pickle.load(f)["features"]
    pca6 = PCA(n_components=10, random_state=42).fit(X_dev_6)
    Z6 = pca6.transform(X_dev_6)
    var6 = pca6.explained_variance_ratio_

    # Dataset 17
    X_test_17 = np.load(p17_dir / "Data/X_test.npy", mmap_mode="r")
    y_test_17 = np.load(p17_dir / "Data/y_test.npy", mmap_mode="r")
    with open(p17_dir / "Data/pipeline.pkl", "rb") as f:
        f17 = pickle.load(f)["features"]

    rng = np.random.default_rng(42)
    fit_idx = rng.choice(len(X_test_17), size=100000, replace=False)
    pca17 = PCA(n_components=10, random_state=42).fit(np.array(X_test_17[fit_idx]))
    var17 = pca17.explained_variance_ratio_

    plot_idx_17 = rng.choice(len(X_test_17), size=5000, replace=False)
    Z17 = pca17.transform(np.array(X_test_17[plot_idx_17]))
    y17 = np.array(y_test_17[plot_idx_17])

    # Sample for dataset 6 plot
    sub6 = rng.choice(len(Z6), size=5000, replace=False)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#ffffff")
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # 1. Scree Comparison
    ax1 = axes[0, 0]
    pcs = np.arange(1, 11)
    ax1.plot(pcs, np.cumsum(var6) * 100, marker="o", color="#1e40af", linewidth=2.5, markersize=6,
             label=f"Dataset 6 (C-MAPSS 24D): PC1+PC2 = {np.sum(var6[:2])*100:.1f}%")
    ax1.plot(pcs, np.cumsum(var17) * 100, marker="s", color="#059669", linewidth=2.5, markersize=6,
             label=f"Dataset 17 (N-CMAPSS 18D): PC1+PC2 = {np.sum(var17[:2])*100:.1f}%")
    ax1.axhline(95, color="#dc2626", linestyle=":", linewidth=1.5, label="95% Information Threshold")
    ax1.set_xlabel("Number of Principal Components", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Variance Explained (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Dimensionality Reduction Efficacy: Cumulative Scree Curves", fontsize=12, fontweight="bold")
    ax1.set_xticks(pcs)
    ax1.set_ylim(60, 102)
    ax1.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Variance Bar Comparison
    ax2 = axes[0, 1]
    width = 0.35
    ax2.bar(pcs[:5] - width/2, var6[:5] * 100, width=width, color="#1e40af", alpha=0.85, label="C-MAPSS (Discrete 24D)")
    ax2.bar(pcs[:5] + width/2, var17[:5] * 100, width=width, color="#059669", alpha=0.85, label="N-CMAPSS (Continuous 18D)")
    ax2.set_xlabel("Principal Component", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Explained Variance (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Variance Concentration per Component (Top 5 PCs)", fontsize=12, fontweight="bold")
    ax2.set_xticks(pcs[:5])
    ax2.set_xticklabels([f"PC{i}" for i in range(1, 6)])
    ax2.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    for i in range(2):
        ax2.text(i+1 - width/2, var6[i]*100 + 1.0, f"{var6[i]*100:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#1e40af")
        ax2.text(i+1 + width/2, var17[i]*100 + 1.0, f"{var17[i]*100:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#059669")

    # 3. C-MAPSS Latent Space
    ax3 = axes[1, 0]
    n6 = y_dev_6[sub6] == 0
    a6 = y_dev_6[sub6] == 1
    ax3.scatter(Z6[sub6][n6, 0], Z6[sub6][n6, 1], c="#2563eb", alpha=0.45, s=12, label="Healthy (RUL > 125)", edgecolors="none")
    ax3.scatter(Z6[sub6][a6, 0], Z6[sub6][a6, 1], c="#dc2626", alpha=0.45, s=12, label="Degraded (RUL \u2264 125)", edgecolors="none")
    plot_confidence_ellipse(Z6[sub6][n6, 0], Z6[sub6][n6, 1], ax3, n_std=1.5, edgecolor="#1d4ed8", linewidth=1.8, linestyle="-")
    plot_confidence_ellipse(Z6[sub6][a6, 0], Z6[sub6][a6, 1], ax3, n_std=1.5, edgecolor="#b91c1c", linewidth=1.8, linestyle="--")
    ax3.set_xlabel(f"PC1 ({var6[0]*100:.1f}% Variance)", fontsize=11, fontweight="bold")
    ax3.set_ylabel(f"PC2 ({var6[1]*100:.1f}% Variance)", fontsize=11, fontweight="bold")
    ax3.set_title("NASA C-MAPSS (Folder 6): Latent Anomaly Separation", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. N-CMAPSS Latent Space
    ax4 = axes[1, 1]
    n17 = y17 == 0
    a17 = y17 == 1
    ax4.scatter(Z17[n17, 0], Z17[n17, 1], c="#2563eb", alpha=0.45, s=12, label="Nominal Flight ($h_s = 1.0$)", edgecolors="none")
    ax4.scatter(Z17[a17, 0], Z17[a17, 1], c="#dc2626", alpha=0.45, s=12, label="Degraded Regime ($h_s = 0.0$)", edgecolors="none")
    plot_confidence_ellipse(Z17[n17, 0], Z17[n17, 1], ax4, n_std=1.5, edgecolor="#1d4ed8", linewidth=1.8, linestyle="-")
    plot_confidence_ellipse(Z17[a17, 0], Z17[a17, 1], ax4, n_std=1.5, edgecolor="#b91c1c", linewidth=1.8, linestyle="--")
    ax4.set_xlabel(f"PC1 ({var17[0]*100:.1f}% Variance)", fontsize=11, fontweight="bold")
    ax4.set_ylabel(f"PC2 ({var17[1]*100:.1f}% Variance)", fontsize=11, fontweight="bold")
    ax4.set_title("NASA N-CMAPSS (Folder 17): Latent Anomaly Separation", fontsize=12, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # Save to both folders
    for out_dir in [p6_dir / "Output", p17_dir / "Output"]:
        out_path = out_dir / "cross_dataset_pca_comparison.png"
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[Done] Comparative PCA Dashboard saved to: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Unified PCA Feature Diagnostics for NASA Turbofan Datasets")
    parser.add_argument("--dataset", type=str, choices=["all", "6", "17"], default="all",
                        help="Which dataset to process: 6, 17, or all (default)")
    parser.add_argument("--dpi", type=int, default=300, help="Plot DPI (default 300)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.dataset in ["all", "6"]:
        plot_dataset_6_pca(base_dir, dpi=args.dpi)
    if args.dataset in ["all", "17"]:
        plot_dataset_17_pca(base_dir, dpi=args.dpi)
    if args.dataset == "all":
        plot_cross_dataset_comparison(base_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
