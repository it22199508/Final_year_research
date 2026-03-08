from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from insider_gru.config import DataConfig
from insider_gru.data import load_event_csvs


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a starter id->label mapping CSV (labels default to 0).")
    ap.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    ap.add_argument("--train-glob", type=str, default=None, help="Override DataConfig.train_glob")
    ap.add_argument("--test-glob", type=str, default=None, help="Override DataConfig.test_glob")
    ap.add_argument("--out", type=Path, default=Path("labels_template.csv"), help="Output CSV path")
    args = ap.parse_args()

    cfg = DataConfig()
    root = args.dataset_root

    train_glob = str(args.train_glob) if args.train_glob is not None else cfg.train_glob
    test_glob = str(args.test_glob) if args.test_glob is not None else cfg.test_glob

    paths = sorted(root.glob(train_glob)) + sorted(root.glob(test_glob))
    if not paths:
        raise FileNotFoundError("No CSVs matched train/test globs.")

    df = load_event_csvs(paths)
    if cfg.id_col not in df.columns:
        raise KeyError(f"Expected id column '{cfg.id_col}'")

    out = pd.DataFrame({cfg.id_col: df[cfg.id_col].astype(str).unique()})
    out["label"] = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} ids. Fill label as 0/1.")


if __name__ == "__main__":
    main()
