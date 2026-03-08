from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from insider_gru.config import DataConfig
from insider_gru.data import load_event_csvs


@dataclass(frozen=True)
class RuleConfig:
    # For the bundled sample dataset, email sizes are typically in the ~10k-80k range.
    # A low default (e.g., 3000) produces an all-positive label set.
    min_size: int = 40000


def label_email_misuse(df: pd.DataFrame, *, size_col: str, rule: RuleConfig) -> pd.Series:
    size_num = pd.to_numeric(df[size_col], errors="coerce").fillna(0)
    return (size_num > rule.min_size).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate id->label mapping for email misuse: label=1 if size>min_size."
        )
    )
    ap.add_argument(
        "--min-size",
        type=int,
        default=RuleConfig.min_size,
        help="Email size threshold (label=1 if size > min_size)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("labels_from_rules.csv"),
        help="Output CSV path (default: labels_from_rules.csv)",
    )
    args = ap.parse_args()

    rule = RuleConfig(min_size=args.min_size)

    cfg = DataConfig()
    root = cfg.dataset_root

    train_paths = sorted(root.glob(cfg.train_glob))
    test_paths = sorted(root.glob(cfg.test_glob))
    paths = train_paths + test_paths
    if not paths:
        raise FileNotFoundError("No CSVs matched train/test globs in DataConfig.")

    df = load_event_csvs(paths)

    required = {cfg.id_col, cfg.timestamp_col, "size"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for rule labeling: {missing}. Columns: {list(df.columns)}")

    labels = label_email_misuse(df, size_col="size", rule=rule)

    out = pd.DataFrame({cfg.id_col: df[cfg.id_col].astype(str), "label": labels.astype(int)})
    out = out.drop_duplicates(subset=[cfg.id_col], keep="last")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    n_pos = int(out["label"].sum())
    n_total = int(len(out))
    n_neg = n_total - n_pos
    print(f"Wrote {args.out} with {n_total} ids. Positives: {n_pos}. Negatives: {n_neg}.")
    print(f"Rule: size>{rule.min_size}")


if __name__ == "__main__":
    main()
