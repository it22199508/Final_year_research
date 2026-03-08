from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from insider_gru.data import ensure_datetime, load_event_csvs


@dataclass(frozen=True)
class TimeWindowRule:
    start_hour: int = 12
    end_hour: int = 20
    activity: str = "Connect"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build id->label mapping for device login events based on a time-of-day rule. "
            "Default: label=1 for activity='Connect' events between 12:00 and 20:00."
        )
    )
    ap.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    ap.add_argument(
        "--train-glob",
        type=str,
        default="train/test/R*/device-train-data.csv",
        help="Glob under dataset-root to include for labeling",
    )
    ap.add_argument(
        "--test-glob",
        type=str,
        default="train/R*/device-test-data*.csv",
        help="Glob under dataset-root to include for labeling",
    )
    ap.add_argument("--start-hour", type=int, default=12, help="Start hour (0-23), inclusive")
    ap.add_argument("--end-hour", type=int, default=20, help="End hour (0-24), exclusive")
    ap.add_argument(
        "--activity",
        type=str,
        default="Connect",
        help="Device activity considered a login (default: Connect)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("labels_device_time_window.csv"),
        help="Output CSV path (columns: id,label)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    rule = TimeWindowRule(
        start_hour=int(args.start_hour),
        end_hour=int(args.end_hour),
        activity=str(args.activity),
    )

    if not (0 <= rule.start_hour <= 23):
        raise ValueError("--start-hour must be in [0,23]")
    if not (1 <= rule.end_hour <= 24):
        raise ValueError("--end-hour must be in [1,24]")
    if rule.start_hour >= rule.end_hour:
        raise ValueError("--start-hour must be < --end-hour")

    root = Path(args.dataset_root)
    paths = sorted(root.glob(str(args.train_glob))) + sorted(root.glob(str(args.test_glob)))
    if not paths:
        raise FileNotFoundError("No device CSVs matched the provided globs.")

    df = load_event_csvs(paths)

    required = {"id", "date", "user", "pc", "activity"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}. Columns: {list(df.columns)}")

    df = ensure_datetime(df, "date")

    # Normalize activity and apply rule
    act = df["activity"].astype(str).str.strip().str.lower()
    is_login = act == rule.activity.strip().lower()

    hour = df["date"].dt.hour
    in_window = (hour >= rule.start_hour) & (hour < rule.end_hour)

    labels = (is_login & in_window).astype(int)

    out = pd.DataFrame({"id": df["id"].astype(str), "label": labels.astype(int)})
    out = out.drop_duplicates(subset=["id"], keep="last")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    n_pos = int(out["label"].sum())
    print(f"Wrote {args.out} with {len(out)} ids. Positive labels: {n_pos}.")
    print(f"Rule: activity='{rule.activity}' AND {rule.start_hour:02d}:00 <= hour < {rule.end_hour:02d}:00")


if __name__ == "__main__":
    main()
