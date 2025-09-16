#!/usr/bin/env python3
# preprocess.py
# Download/load churn dataset, do basic cleaning, save processed CSV (Mac/Unix friendly).

import argparse
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np

# headless-safe plotting backend; seaborn is optional
import matplotlib
matplotlib.use("Agg")
try:
    import seaborn as sns  # optional EDA
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

def download_kaggle_dataset(dataset: str, raw_dir: Path):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print("Kaggle package not installed. Run: pip install kaggle", file=sys.stderr)
        raise

    # Ensure creds exist
    cred_path = Path("~/.kaggle/kaggle.json").expanduser()
    if not cred_path.exists() and not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        raise FileNotFoundError(
            "Kaggle credentials not found. Put kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME/KAGGLE_KEY."
        )

    api = KaggleApi()
    api.authenticate()
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[KAGGLE] Downloading {dataset} into {raw_dir} ...")
    api.dataset_download_files(dataset, path=str(raw_dir), unzip=True)
    print("[KAGGLE] Download complete.")

def pick_csv(raw_dir: Path, prefer_keyword="churn"):
    csvs = list(raw_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    # prefer a CSV that contains 'churn' in the filename
    for p in csvs:
        if prefer_keyword.lower() in p.name.lower():
            return p
    return csvs[0]

def basic_clean(df: pd.DataFrame):
    # Try to standardize the target column to 'churn' (0/1)
    target_candidates = ["Churn", "churn", "Exited"]
    target = None
    for c in target_candidates:
        if c in df.columns:
            target = c
            break
    if target is None:
        raise ValueError(f"Could not find a churn target column among {target_candidates}. Found: {list(df.columns)}")

    # Map yes/no, string to 0/1 if needed
    if df[target].dtype == object:
        df[target] = df[target].astype(str).str.strip().str.lower().map({"yes":1, "no":0, "true":1, "false":0})
    # If still not numeric, try to coerce
    if not np.issubdtype(df[target].dtype, np.number):
        df[target] = pd.to_numeric(df[target], errors="coerce")

    # Common column cleanups (if present)
    for col in ["TotalCharges", "Total_Charges", "total_charges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where target is NaN after mapping
    before = len(df)
    df = df.dropna(subset=[target])
    after = len(df)
    if after < before:
        print(f"[CLEAN] Dropped {before - after} rows with missing target values.")

    # Simple example: rename target to 'churn'
    if target != "churn":
        df = df.rename(columns={target: "churn"})

    # Optional: basic EDA plot if seaborn is available
    if _HAS_SNS and "churn" in df.columns:
        ax = sns.countplot(x="churn", data=df)
        fig = ax.get_figure()
        Path("results/eda").mkdir(parents=True, exist_ok=True)
        fig.savefig("results/eda/churn_distribution.png", dpi=180)
        fig.clf()

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle", "local"], default="kaggle",
                    help="Where to load data from.")
    ap.add_argument("--kaggle_dataset", default="royjafari/customer-churn",
                    help="Kaggle dataset slug (owner/dataset).")
    ap.add_argument("--raw_dir", default="data/raw", help="Directory to store raw files.")
    ap.add_argument("--input_csv", default="", help="If --source=local, path to local CSV.")
    ap.add_argument("--output", default="data/processed/telecom_churn_processed.csv",
                    help="Output processed CSV path.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "kaggle":
        download_kaggle_dataset(args.kaggle_dataset, raw_dir)
        csv_path = pick_csv(raw_dir, prefer_keyword="churn")
    else:
        if not args.input_csv:
            raise ValueError("When --source=local, you must provide --input_csv")
        csv_path = Path(args.input_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Local CSV not found: {csv_path}")

    print(f"[LOAD] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    df_clean = basic_clean(df)
    df_clean.to_csv(out_csv, index=False)
    print(f"[OK] Wrote processed CSV → {out_csv.resolve()}")

if __name__ == "__main__":
    main()
