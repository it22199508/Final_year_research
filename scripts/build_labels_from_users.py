from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from insider_gru.config import DataConfig
from insider_gru.data import load_event_csvs


def read_users_list(path: Path) -> set[str]:
    users: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        users.add(s)
    return users


def pick_user_col(df: pd.DataFrame) -> str:
    # Email data: "from" is typical. Device/HTTP: "user" is typical.
    for c in ("from", "user"):
        if c in df.columns:
            return c
    raise KeyError(f"Could not find a user column in CSV. Columns: {list(df.columns)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build id->label mapping from a list of insider users.")
    ap.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    ap.add_argument("--train-glob", type=str, default=None, help="Override DataConfig.train_glob")
    ap.add_argument("--test-glob", type=str, default=None, help="Override DataConfig.test_glob")
    ap.add_argument(
        "--users-file",
        type=Path,
        required=True,
        help="Text file with one user/email per line. Lines starting with # are ignored.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("labels_from_users.csv"),
        help="Output CSV path (default: labels_from_users.csv)",
    )
    args = ap.parse_args()

    cfg = DataConfig()
    root = args.dataset_root

    train_glob = str(args.train_glob) if args.train_glob is not None else cfg.train_glob
    test_glob = str(args.test_glob) if args.test_glob is not None else cfg.test_glob

    train_paths = sorted(root.glob(train_glob))
    test_paths = sorted(root.glob(test_glob))
    paths = train_paths + test_paths
    if not paths:
        raise FileNotFoundError("No CSVs matched train/test globs in DataConfig.")

    insider_users = read_users_list(args.users_file)
    if not insider_users:
        raise ValueError("Users file is empty after filtering comments/blank lines.")

    df = load_event_csvs(paths)

    if cfg.id_col not in df.columns:
        raise KeyError(f"Expected id column '{cfg.id_col}', got columns: {list(df.columns)}")

    user_col = pick_user_col(df)
    user_series = df[user_col].astype(str)

    # Label each event by whether its user is in insider list
    labels = user_series.isin(insider_users).astype(int)

    out = pd.DataFrame({cfg.id_col: df[cfg.id_col].astype(str), "label": labels.astype(int)})
    # Keep unique ids (id should already be unique, but just in case)
    out = out.drop_duplicates(subset=[cfg.id_col], keep="last")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    n_pos = int(out["label"].sum())
    print(f"Wrote {args.out} with {len(out)} ids. Positive labels: {n_pos}.")
    print(f"Detected user column: {user_col}")


if __name__ == "__main__":
    main()
