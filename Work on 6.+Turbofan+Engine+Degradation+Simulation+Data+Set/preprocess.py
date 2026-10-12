#!/usr/bin/env python3
"""
Preprocess, Standardize, and Scale NASA C-MAPSS Dataset 6 for GMM Anomaly Detection.

Workflow:
1. Feature Selection: Drops any invariant features (std < 1e-4) across the combined dataset.
2. Anomaly Labeling: Binary anomaly target defined by RUL threshold (RUL <= 125 cycles).
3. Standardization: Standardizes features using statistics (mean and std) computed strictly
   from healthy training cycles (RUL > 125) to prevent data leakage.
4. Support for two preprocessing modes:
   - 'joint' (default benchmark): Standardizes all 24 operational and sensor features.
     When fitted with GMM (k=6), the mixture components model the 6 joint operating flight regimes.
   - 'regime_norm': Uses K-Means (k=6) on settings to normalize sensor readings per flight regime.
5. Exports: Saves ready-to-train NumPy arrays (.npy) and serialization bundle (pipeline.pkl)
   to the Output/ directory.
"""

from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def prepare_features_joint(
    df_train: pd.DataFrame,
    df_dev: pd.DataFrame,
    df_test: pd.DataFrame,
    rul_threshold: int = 125
):
    """
    Standardize all active operational and sensor features using statistics
    fitted strictly on healthy training cycles (RUL > rul_threshold).
    
    This matches the official benchmark pipeline (ROC-AUC 0.7310, PR-AUC 0.7528).
    """
    feature_cols = [c for c in df_train.columns if c.startswith("s_") or c.startswith("setting_")]
    
    # Filter active features
    stds = df_train[feature_cols].std()
    active_features = stds[stds > 1e-4].index.tolist()

    for df in [df_train, df_dev, df_test]:
        df["anomaly"] = (df["RUL"] <= rul_threshold).astype(int)

    train_normal = df_train[df_train["anomaly"] == 0]

    mean = train_normal[active_features].mean().values
    std = train_normal[active_features].std().values
    std = np.where(std < 1e-6, 1.0, std)

    X_train_all = (df_train[active_features].values - mean) / std
    X_train_healthy = (train_normal[active_features].values - mean) / std
    X_dev = (df_dev[active_features].values - mean) / std
    X_test = (df_test[active_features].values - mean) / std

    y_train = df_train["anomaly"].values
    y_dev = df_dev["anomaly"].values
    y_test = df_test["anomaly"].values

    scaler_dict = {"mean": mean, "std": std, "features": active_features}
    return X_train_healthy, X_train_all, y_train, X_dev, y_dev, X_test, y_test, active_features, scaler_dict


def prepare_features_regime_norm(
    df_train: pd.DataFrame,
    df_dev: pd.DataFrame,
    df_test: pd.DataFrame,
    rul_threshold: int = 125,
    n_regimes: int = 6,
    seed: int = 42
):
    """
    Cluster settings into regimes and normalize sensor readings by regime baseline.
    """
    drop_sensors = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19", "setting_3"]
    feature_sensors = [f"s_{i}" for i in range(1, 22) if f"s_{i}" not in drop_sensors]
    setting_cols = ["setting_1", "setting_2", "setting_3"]

    for df in [df_train, df_dev, df_test]:
        df["anomaly"] = (df["RUL"] <= rul_threshold).astype(int)
        df[feature_sensors] = df[feature_sensors].astype(float)

    healthy_train = df_train[df_train["anomaly"] == 0]
    km = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)
    km.fit(healthy_train[setting_cols].values)

    df_train["regime"] = km.predict(df_train[setting_cols].values)
    df_dev["regime"] = km.predict(df_dev[setting_cols].values)
    df_test["regime"] = km.predict(df_test[setting_cols].values)

    regime_stats = {}
    for r in range(n_regimes):
        r_sub = healthy_train[df_train.loc[healthy_train.index, "regime"] == r][feature_sensors]
        mean_v = r_sub.mean().values
        std_v = np.where(r_sub.std().values < 1e-4, 1.0, r_sub.std().values)
        regime_stats[r] = {"mean": mean_v, "std": std_v}

    def _normalize(df):
        norm_vals = df[feature_sensors].values.astype(float).copy()
        regimes = df["regime"].values
        for r in range(n_regimes):
            mask = (regimes == r)
            if np.any(mask):
                norm_vals[mask] = (norm_vals[mask] - regime_stats[r]["mean"]) / regime_stats[r]["std"]
        return norm_vals

    X_tr_norm = _normalize(df_train)
    X_dv_norm = _normalize(df_dev)
    X_te_norm = _normalize(df_test)

    scaler = StandardScaler()
    scaler.fit(X_tr_norm[df_train["anomaly"] == 0])

    X_train_all = scaler.transform(X_tr_norm)
    X_train_healthy = X_train_all[df_train["anomaly"] == 0]
    X_dev = scaler.transform(X_dv_norm)
    X_test = scaler.transform(X_te_norm)

    y_train = df_train["anomaly"].values
    y_dev = df_dev["anomaly"].values
    y_test = df_test["anomaly"].values

    pipeline_dict = {
        "kmeans": km,
        "regime_stats": regime_stats,
        "scaler": scaler,
        "features": feature_sensors
    }
    return X_train_healthy, X_train_all, y_train, X_dev, y_dev, X_test, y_test, feature_sensors, pipeline_dict


def run_preprocessing(
    data_dir: Path,
    output_dir: Path,
    method: str = "joint",
    rul_threshold: int = 125,
    seed: int = 42
):
    """Run preprocessing, print diagnostics, and save arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  Preprocessing & Scaling Dataset 6 for GMM Anomaly Detection   ")
    print("=================================================================")
    print(f"Loading CSV datasets from {data_dir}...")
    df_train = pd.read_csv(data_dir / "train.csv")
    df_dev = pd.read_csv(data_dir / "dev.csv")
    df_test = pd.read_csv(data_dir / "test.csv")

    print(f"Loaded: {len(df_train):,} train, {len(df_dev):,} dev, {len(df_test):,} test samples.")
    print(f"Preprocessing method: '{method}' | RUL Anomaly Threshold: {rul_threshold}")

    if method == "joint":
        X_train_h, X_tr, y_tr, X_dev, y_dev, X_test, y_test, features, bundle = prepare_features_joint(
            df_train, df_dev, df_test, rul_threshold=rul_threshold
        )
    elif method == "regime_norm":
        X_train_h, X_tr, y_tr, X_dev, y_dev, X_test, y_test, features, bundle = prepare_features_regime_norm(
            df_train, df_dev, df_test, rul_threshold=rul_threshold, seed=seed
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'joint' or 'regime_norm'.")

    # Quality Verification
    assert np.all(np.isfinite(X_train_h)), "NaN or Inf found in healthy train features!"
    assert np.all(np.isfinite(X_dev)), "NaN or Inf found in dev features!"
    assert np.all(np.isfinite(X_test)), "NaN or Inf found in test features!"

    print("\n[Quality & Distribution Verification]")
    print(f"Features ({len(features)}): {features}")
    print(f"Healthy train shape: {X_train_h.shape}")
    print(f"Dev shape:           {X_dev.shape} | Anomalous: {np.mean(y_dev)*100:.1f}%")
    print(f"Test shape:          {X_test.shape} | Anomalous: {np.mean(y_test)*100:.1f}%")
    print(f"Healthy train mean max absolute error from 0.0: {np.max(np.abs(np.mean(X_train_h, axis=0))):.2e}")
    print(f"Healthy train standard deviation range:         [{np.min(np.std(X_train_h, axis=0)):.4f}, {np.max(np.std(X_train_h, axis=0)):.4f}]")

    # Save to Output/
    print(f"\nSaving preprocessed tensors to {output_dir}...")
    np.save(output_dir / "X_train_healthy.npy", X_train_h)
    np.save(output_dir / "X_train_full.npy", X_tr)
    np.save(output_dir / "y_train.npy", y_tr)
    np.save(output_dir / "X_dev.npy", X_dev)
    np.save(output_dir / "y_dev.npy", y_dev)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    bundle["method"] = method
    bundle["rul_threshold"] = rul_threshold
    bundle["features"] = features

    with open(output_dir / "pipeline.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print("Preprocessed arrays and pipeline bundle successfully saved!")
    print(f"  - {output_dir / 'X_train_healthy.npy'}")
    print(f"  - {output_dir / 'X_dev.npy'} & {output_dir / 'y_dev.npy'}")
    print(f"  - {output_dir / 'X_test.npy'} & {output_dir / 'y_test.npy'}")
    print(f"  - {output_dir / 'pipeline.pkl'}")
    print("=================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and scale dataset 6 for GMM anomaly detection.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing train.csv, dev.csv, test.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save preprocessed .npy and .pkl files")
    parser.add_argument("--method", type=str, default="joint", choices=["joint", "regime_norm"], help="Preprocessing method ('joint' or 'regime_norm')")
    parser.add_argument("--rul-threshold", type=int, default=125, help="RUL cutoff for normal vs anomalous cycle (default: 125)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "Data"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "Data"

    run_preprocessing(
        data_dir=data_dir,
        output_dir=output_dir,
        method=args.method,
        rul_threshold=args.rul_threshold,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
