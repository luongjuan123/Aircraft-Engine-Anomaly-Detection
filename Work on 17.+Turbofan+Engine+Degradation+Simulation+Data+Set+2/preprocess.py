#!/usr/bin/env python3
"""
Preprocess, Standardize, and Scale NASA N-CMAPSS Dataset 17 for GMM Anomaly Detection.

Workflow:
1. Data Ingestion: Reads raw HDF5 dataset (default: N-CMAPSS_DS02-006.h5) containing
   scenario descriptors (W), physical sensor measurements (X_s), virtual sensors (X_v),
   auxiliary info (A: unit, cycle, Fc, hs), and remaining useful life (Y).
2. Anomaly Ground Truth: Binary anomaly target defined strictly by the ground-truth
   health state flag hs (hs = 1 for nominal operation, hs = 0 for degraded/faulty operation).
   anomaly = (hs == 0).astype(int).
3. Standardization: Standardizes features using statistics (mean and std) computed strictly
   from healthy training cycles (hs == 1) on the Dev set to prevent data leakage.
4. Feature Modes:
   - 'joint' (default benchmark): Standardizes all 4 operational conditions (W) and
     14 physical sensor measurements (X_s) = 18 features. When fitted with GMM (k=6),
     the mixture components model the multi-regime commercial flight envelopes.
   - 'sensors': Standardizes the 14 physical sensor measurements (X_s) alone.
   - 'all': Standardizes W (4), X_s (14), and virtual sensors X_v (14) = 32 features.
5. Exports: Saves ready-to-train NumPy arrays (.npy) and serialization bundle (pipeline.pkl)
   to the Data/ directory.
"""

from pathlib import Path
import argparse
import pickle
import h5py
import numpy as np


def load_ncmaps_hdf5(file_path: Path):
    """Load scenario, sensor, auxiliary, and target arrays from an N-CMAPSS HDF5 file."""
    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found at: {file_path}")

    with h5py.File(file_path, "r") as hdf:
        W_dev = np.asarray(hdf["W_dev"], dtype=np.float32)
        X_s_dev = np.asarray(hdf["X_s_dev"], dtype=np.float32)
        X_v_dev = np.asarray(hdf["X_v_dev"], dtype=np.float32) if "X_v_dev" in hdf else None
        A_dev = np.asarray(hdf["A_dev"], dtype=np.float32)
        Y_dev = np.asarray(hdf["Y_dev"], dtype=np.float32)

        W_test = np.asarray(hdf["W_test"], dtype=np.float32)
        X_s_test = np.asarray(hdf["X_s_test"], dtype=np.float32)
        X_v_test = np.asarray(hdf["X_v_test"], dtype=np.float32) if "X_v_test" in hdf else None
        A_test = np.asarray(hdf["A_test"], dtype=np.float32)
        Y_test = np.asarray(hdf["Y_test"], dtype=np.float32)

        def decode_vars(key):
            if key not in hdf:
                return []
            return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in hdf[key][:]]

        W_var = decode_vars("W_var")
        X_s_var = decode_vars("X_s_var")
        X_v_var = decode_vars("X_v_var")
        A_var = decode_vars("A_var")

    return {
        "W_dev": W_dev,
        "X_s_dev": X_s_dev,
        "X_v_dev": X_v_dev,
        "A_dev": A_dev,
        "Y_dev": Y_dev,
        "W_test": W_test,
        "X_s_test": X_s_test,
        "X_v_test": X_v_test,
        "A_test": A_test,
        "Y_test": Y_test,
        "W_var": W_var,
        "X_s_var": X_s_var,
        "X_v_var": X_v_var,
        "A_var": A_var,
    }


def prepare_features(
    data: dict,
    method: str = "joint",
    subsample_train: int = 50000,
    stride: int = 1,
    seed: int = 42
):
    """
    Standardize features using zero-leakage statistics computed exclusively on
    healthy training cycles (hs == 1) from the Dev partition.
    """
    if method == "joint":
        features = data["W_var"] + data["X_s_var"]
        feat_dev = np.hstack([data["W_dev"], data["X_s_dev"]])
        feat_test = np.hstack([data["W_test"], data["X_s_test"]])
    elif method == "sensors":
        features = data["X_s_var"]
        feat_dev = data["X_s_dev"]
        feat_test = data["X_s_test"]
    elif method == "all":
        if data["X_v_dev"] is None:
            raise ValueError("Virtual sensors (X_v) not available in this dataset file.")
        features = data["W_var"] + data["X_s_var"] + data["X_v_var"]
        feat_dev = np.hstack([data["W_dev"], data["X_s_dev"], data["X_v_dev"]])
        feat_test = np.hstack([data["W_test"], data["X_s_test"], data["X_v_test"]])
    else:
        raise ValueError(f"Unknown preprocessing method '{method}'. Choose 'joint', 'sensors', or 'all'.")

    # A[:, 3] is health state (hs): 1 = nominal, 0 = degraded
    hs_dev = data["A_dev"][:, 3]
    hs_test = data["A_test"][:, 3]

    y_dev_full = (hs_dev == 0).astype(np.int8)
    y_test_full = (hs_test == 0).astype(np.int8)

    # Identify healthy training samples (hs == 1, i.e. y == 0)
    healthy_idx = np.where(y_dev_full == 0)[0]
    if len(healthy_idx) == 0:
        raise ValueError("No healthy samples (hs == 1) found in Dev partition!")

    # Zero-leakage statistics computed strictly on nominal/healthy cycles
    mean = np.mean(feat_dev[healthy_idx], axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(feat_dev[healthy_idx], axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    # Subsample healthy training data for fast, accurate EM estimation
    rng = np.random.default_rng(seed)
    n_train = min(subsample_train, len(healthy_idx))
    chosen_train_idx = rng.choice(healthy_idx, size=n_train, replace=False)
    X_train_healthy = (feat_dev[chosen_train_idx] - mean) / std

    # Apply stride to dev and test if requested
    if stride > 1:
        dev_slice = slice(None, None, stride)
        test_slice = slice(None, None, stride)
    else:
        dev_slice = slice(None)
        test_slice = slice(None)

    X_dev = (feat_dev[dev_slice] - mean) / std
    y_dev = y_dev_full[dev_slice]
    A_dev = data["A_dev"][dev_slice]
    Y_dev = data["Y_dev"][dev_slice]

    X_test = (feat_test[test_slice] - mean) / std
    y_test = y_test_full[test_slice]
    A_test = data["A_test"][test_slice]
    Y_test = data["Y_test"][test_slice]

    pipeline_bundle = {
        "method": method,
        "features": features,
        "mean": mean,
        "std": std,
        "subsample_train": n_train,
        "stride": stride,
        "units_dev": np.unique(data["A_dev"][:, 0]).tolist(),
        "units_test": np.unique(data["A_test"][:, 0]).tolist(),
        "total_dev_samples": len(feat_dev),
        "total_test_samples": len(feat_test),
        "dev_healthy_samples": len(healthy_idx),
    }

    return (
        X_train_healthy,
        X_dev,
        y_dev,
        A_dev,
        Y_dev,
        X_test,
        y_test,
        A_test,
        Y_test,
        pipeline_bundle,
    )


def run_preprocessing(
    data_file: Path,
    output_dir: Path,
    method: str = "joint",
    subsample_train: int = 50000,
    stride: int = 1,
    seed: int = 42
):
    """Run end-to-end preprocessing, distribution diagnostics, and disk export."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  Preprocessing & Scaling Dataset 17 (N-CMAPSS) for GMM          ")
    print("=================================================================")
    print(f"Loading HDF5 dataset from: {data_file}")
    data = load_ncmaps_hdf5(data_file)

    dev_units = np.unique(data["A_dev"][:, 0]).astype(int)
    test_units = np.unique(data["A_test"][:, 0]).astype(int)

    print(f"Loaded {len(data['A_dev']):,} Dev points across {len(dev_units)} units: {dev_units.tolist()}")
    print(f"Loaded {len(data['A_test']):,} Test points across {len(test_units)} units: {test_units.tolist()}")
    print(f"Preprocessing method: '{method}' | Subsample Train: {subsample_train:,} | Stride: {stride}")

    (
        X_train_h,
        X_dev,
        y_dev,
        A_dev,
        Y_dev,
        X_test,
        y_test,
        A_test,
        Y_test,
        pipeline_bundle,
    ) = prepare_features(
        data=data,
        method=method,
        subsample_train=subsample_train,
        stride=stride,
        seed=seed,
    )

    # Distribution and Quality Verification
    assert np.all(np.isfinite(X_train_h)), "NaN or Inf found in healthy train features!"
    assert np.all(np.isfinite(X_dev)), "NaN or Inf found in dev features!"
    assert np.all(np.isfinite(X_test)), "NaN or Inf found in test features!"

    print("\n[Quality & Distribution Verification]")
    print(f"Active Features ({len(pipeline_bundle['features'])}): {pipeline_bundle['features']}")
    print(f"Healthy Train Shape: {X_train_h.shape}")
    print(f"Dev Feature Shape:   {X_dev.shape} | Anomalous (hs=0): {np.mean(y_dev)*100:.1f}%")
    print(f"Test Feature Shape:  {X_test.shape} | Anomalous (hs=0): {np.mean(y_test)*100:.1f}%")
    print(f"Healthy Train mean max absolute deviation from 0.0: {np.max(np.abs(np.mean(X_train_h, axis=0))):.2e}")
    print(f"Healthy Train std range: [{np.min(np.std(X_train_h, axis=0)):.4f}, {np.max(np.std(X_train_h, axis=0)):.4f}]")

    # Save preprocessed arrays
    print(f"\nSaving preprocessed tensors to {output_dir}...")
    np.save(output_dir / "X_train_healthy.npy", X_train_h)
    np.save(output_dir / "X_dev.npy", X_dev)
    np.save(output_dir / "y_dev.npy", y_dev)
    np.save(output_dir / "A_dev.npy", A_dev)
    np.save(output_dir / "Y_dev.npy", Y_dev)

    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    np.save(output_dir / "A_test.npy", A_test)
    np.save(output_dir / "Y_test.npy", Y_test)

    with open(output_dir / "pipeline.pkl", "wb") as f:
        pickle.dump(pipeline_bundle, f)

    print("Preprocessed arrays and pipeline bundle successfully saved:")
    print(f"  - {output_dir / 'X_train_healthy.npy'}")
    print(f"  - {output_dir / 'X_dev.npy'} & {output_dir / 'y_dev.npy'}")
    print(f"  - {output_dir / 'X_test.npy'} & {output_dir / 'y_test.npy'}")
    print(f"  - {output_dir / 'pipeline.pkl'}")
    print("=================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and scale NASA N-CMAPSS (Dataset 17) for GMM anomaly detection.")
    parser.add_argument("--data-file", type=str, default=None, help="Path to N-CMAPSS HDF5 file (default: Data/N-CMAPSS_DS02-006.h5)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save preprocessed .npy and .pkl files")
    parser.add_argument("--method", type=str, default="joint", choices=["joint", "sensors", "all"], help="Feature mode ('joint', 'sensors', or 'all')")
    parser.add_argument("--subsample-train", type=int, default=50000, help="Number of healthy cycles for training GMM (default: 50000)")
    parser.add_argument("--stride", type=int, default=1, help="Downsampling stride for dev/test arrays (default: 1 for full resolution)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    default_data_file = base_dir / "Data" / "N-CMAPSS_DS02-006.h5"
    data_file = Path(args.data_file) if args.data_file else default_data_file
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "Data"

    run_preprocessing(
        data_file=data_file,
        output_dir=output_dir,
        method=args.method,
        subsample_train=args.subsample_train,
        stride=args.stride,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
