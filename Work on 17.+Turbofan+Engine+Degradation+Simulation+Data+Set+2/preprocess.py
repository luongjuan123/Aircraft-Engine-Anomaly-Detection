#!/usr/bin/env python3
"""
Preprocess Folder 17 N-CMAPSS Datasets.

Combines ALL valid N-CMAPSS datasets (DS01 to DS08) into ONE single dataset,
randomly shuffles all samples, and splits into 80% train, 10% dev, 10% test.
Outputs exactly 3 CSV files in Processed data/:
  - train.csv
  - dev.csv
  - test.csv
"""

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import h5py


def load_and_combine_all_n_cmapss(raw_dir: Path, stride: int = 10):
    files = sorted(raw_dir.glob("*.h5"))
    dfs = []
    total_loaded = 0

    col_names = None

    for f in files:
        try:
            with h5py.File(f, "r") as hdf:
                if col_names is None:
                    A_var = [x.decode() if isinstance(x, bytes) else str(x) for x in hdf["A_var"]]
                    W_var = [x.decode() if isinstance(x, bytes) else str(x) for x in hdf["W_var"]]
                    X_s_var = [x.decode() if isinstance(x, bytes) else str(x) for x in hdf["X_s_var"]]
                    col_names = A_var + W_var + X_s_var + ["RUL", "dataset_id"]

                stem = f.stem.replace("N-CMAPSS_", "").split("-")[0]

                A_dev = hdf["A_dev"][::stride]
                W_dev = hdf["W_dev"][::stride]
                X_s_dev = hdf["X_s_dev"][::stride]
                Y_dev = hdf["Y_dev"][::stride]

                A_test = hdf["A_test"][::stride]
                W_test = hdf["W_test"][::stride]
                X_s_test = hdf["X_s_test"][::stride]
                Y_test = hdf["Y_test"][::stride]

            A = np.vstack([A_dev, A_test])
            W = np.vstack([W_dev, W_test])
            X_s = np.vstack([X_s_dev, X_s_test])
            Y = np.vstack([Y_dev, Y_test])

            n_samples = len(A)
            # Create dataset_id column
            ds_col = np.full((n_samples, 1), stem, dtype=object)

            matrix = np.hstack([A, W, X_s, Y])
            df = pd.DataFrame(matrix, columns=col_names[:-1])
            df["dataset_id"] = stem
            df["unit"] = df["unit"].astype(int)
            df["cycle"] = df["cycle"].astype(int)
            df["RUL"] = df["RUL"].astype(int)

            dfs.append(df)
            total_loaded += n_samples
            print(f"Loaded {stem} ({f.name}): {n_samples:,} samples")

        except Exception as e:
            print(f"Warning: Skipping {f.name} due to read error: {e}")

    print(f"\nTotal combined samples across all N-CMAPSS datasets: {total_loaded:,}")
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def shuffle_and_split(df: pd.DataFrame, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, random_state=42):
    assert np.isclose(train_ratio + dev_ratio + test_ratio, 1.0)

    print("\nRandomly shuffling all samples across combined dataset...")
    shuffled_df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_total = len(shuffled_df)

    n_train = int(np.round(train_ratio * n_total))
    n_dev = int(np.round(dev_ratio * n_total))
    n_test = n_total - n_train - n_dev

    df_train = shuffled_df.iloc[:n_train].reset_index(drop=True)
    df_dev = shuffled_df.iloc[n_train:n_train + n_dev].reset_index(drop=True)
    df_test = shuffled_df.iloc[n_train + n_dev:].reset_index(drop=True)

    return df_train, df_dev, df_test


def main():
    parser = argparse.ArgumentParser(description="Preprocess and combine all N-CMAPSS datasets (Folder 17)")
    parser.add_argument("--stride", type=int, default=10, help="Temporal decimation stride (default=10 for 0.1 Hz)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2" / "17. Turbofan Engine Degradation Simulation Data Set 2" / "data_set"
    output_dir = base_dir / "Work on 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2" / "Processed data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Combining all datasets in Folder 17 (stride={args.stride}) ===")
    t0 = time.time()
    df_combined = load_and_combine_all_n_cmapss(raw_dir, stride=args.stride)
    print(f"Loaded in {time.time() - t0:.2f}s")

    t1 = time.time()
    df_train, df_dev, df_test = shuffle_and_split(df_combined, 0.8, 0.1, 0.1, random_state=args.seed)
    print(f"Shuffled and split in {time.time() - t1:.2f}s")

    train_path = output_dir / "train.csv"
    dev_path = output_dir / "dev.csv"
    test_path = output_dir / "test.csv"

    print("Writing CSV files...")
    t2 = time.time()
    df_train.to_csv(train_path, index=False)
    print(f"Saved {train_path} ({len(df_train):,} rows)")
    df_dev.to_csv(dev_path, index=False)
    print(f"Saved {dev_path} ({len(df_dev):,} rows)")
    df_test.to_csv(test_path, index=False)
    print(f"Saved {test_path} ({len(df_test):,} rows)")
    print(f"CSV export completed in {time.time() - t2:.2f}s")

    n_total = len(df_combined)
    print(f"\nDone! Exactly 3 CSV files created in {output_dir}:")
    print(f"  train.csv: {len(df_train):,} rows ({len(df_train)/n_total*100:.1f}%)")
    print(f"  dev.csv:   {len(df_dev):,} rows ({len(df_dev)/n_total*100:.1f}%)")
    print(f"  test.csv:  {len(df_test):,} rows ({len(df_test)/n_total*100:.1f}%)")


if __name__ == "__main__":
    main()
