#!/usr/bin/env python3
"""
Preprocess Folder 6 C-MAPSS Datasets.

Combines ALL datasets (FD001, FD002, FD003, FD004) - mixing both train and test files,
computes RUL, randomly shuffles the entire data, and splits into 80% train, 10% dev, 10% test.
Outputs exactly 3 CSV files in Processed data/:
  - train.csv
  - dev.csv
  - test.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

COLUMN_NAMES = [
    "unit_nr",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
    "s_1", "s_2", "s_3", "s_4", "s_5",
    "s_6", "s_7", "s_8", "s_9", "s_10",
    "s_11", "s_12", "s_13", "s_14", "s_15",
    "s_16", "s_17", "s_18", "s_19", "s_20", "s_21"
]

DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def load_and_combine_all(raw_dir: Path) -> pd.DataFrame:
    all_dfs = []
    
    for name in DATASETS:
        train_file = raw_dir / f"train_{name}.txt"
        test_file = raw_dir / f"test_{name}.txt"
        rul_file = raw_dir / f"RUL_{name}.txt"

        # 1. Train data
        df_tr = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
        max_c_tr = df_tr.groupby("unit_nr")["time_cycles"].transform("max")
        df_tr["RUL"] = max_c_tr - df_tr["time_cycles"]
        df_tr["dataset_id"] = name
        df_tr["source"] = "train"

        # 2. Test data
        df_te = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
        rul_series = pd.read_csv(rul_file, sep=r"\s+", header=None)[0]
        test_rul_map = {u: rul_series.iloc[u - 1] for u in range(1, len(rul_series) + 1)}
        max_c_te = df_te.groupby("unit_nr")["time_cycles"].transform("max")
        final_rul = df_te["unit_nr"].map(test_rul_map)
        df_te["RUL"] = (max_c_te + final_rul) - df_te["time_cycles"]
        df_te["dataset_id"] = name
        df_te["source"] = "test"

        # Offset test unit IDs so all units within dataset are distinct
        max_tr_unit = df_tr["unit_nr"].max()
        df_te["unit_nr"] = df_te["unit_nr"] + max_tr_unit

        all_dfs.extend([df_tr, df_te])
        print(f"Loaded {name}: {len(df_tr):,} train cycles + {len(df_te):,} test cycles = {len(df_tr) + len(df_te):,} total")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined samples across all Folder 6 datasets: {len(combined_df):,}")
    return combined_df


def shuffle_and_split(df: pd.DataFrame, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, random_state=42):
    assert np.isclose(train_ratio + dev_ratio + test_ratio, 1.0)
    
    # Randomly shuffle all samples
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
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "6.+Turbofan+Engine+Degradation+Simulation+Data+Set" / "6. Turbofan Engine Degradation Simulation Data Set" / "CMAPSSData"
    output_dir = base_dir / "Work on 6.+Turbofan+Engine+Degradation+Simulation+Data+Set" / "Processed data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Combining all datasets in Folder 6 ===")
    df_combined = load_and_combine_all(raw_dir)

    print("\n=== Randomly shuffling and splitting 80% / 10% / 10% ===")
    df_train, df_dev, df_test = shuffle_and_split(df_combined, 0.8, 0.1, 0.1, random_state=42)

    train_path = output_dir / "train.csv"
    dev_path = output_dir / "dev.csv"
    test_path = output_dir / "test.csv"

    print("Saving to CSV files...")
    df_train.to_csv(train_path, index=False)
    df_dev.to_csv(dev_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"\nDone! Only 3 CSV files created in {output_dir}:")
    print(f"  train.csv: {len(df_train):,} rows ({len(df_train)/len(df_combined)*100:.1f}%)")
    print(f"  dev.csv:   {len(df_dev):,} rows ({len(df_dev)/len(df_combined)*100:.1f}%)")
    print(f"  test.csv:  {len(df_test):,} rows ({len(df_test)/len(df_combined)*100:.1f}%)")


if __name__ == "__main__":
    main()
