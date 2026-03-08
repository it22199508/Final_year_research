"""Reusable streaming helpers for CSV -> live event simulation.

This module intentionally stays small and generic: it does not depend on
Streamlit and it does not run the ML pipeline directly.

The goal is to support a simple workflow:
1) Load a CSV source.
2) Yield events one-by-one in row order with a configurable delay.
3) Write emitted events / generated alerts to JSONL outputs.

It is reusable for both email and login/device event simulations.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


def _to_jsonable(value: Any) -> Any:
    """Convert common pandas/numpy scalars into JSON-friendly Python types.

    This keeps JSONL writing stable when DataFrames contain numpy dtypes.
    """

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, (pd.Timestamp, datetime)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
    except Exception:
        pass

    return str(value)


def load_event_source(path: str | Path) -> pd.DataFrame:
    """Load an event CSV safely.

    This project includes CERT-style CSVs where each line may be wrapped in
    quotes (including the header). A naive `pd.read_csv` can mis-parse such
    files into a single column.

    This loader:
    - raises a clear error if the file is missing
    - raises a clear error if the file is empty
    - attempts a normal read first, then falls back to a robust mode

    Args:
        path: CSV path.

    Returns:
        DataFrame containing the CSV content.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is empty or cannot be parsed.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")

    try:
        if p.stat().st_size == 0:
            raise ValueError(f"CSV file is empty: {p}")
    except OSError:
        # If stat fails, still attempt to read and validate emptiness.
        pass

    # First attempt: normal pandas parsing.
    try:
        df = pd.read_csv(p)
    except Exception as e:
        df = None
        first_error = e
    else:
        first_error = None

    # If we got a valid-looking dataframe, return it.
    if isinstance(df, pd.DataFrame) and df.shape[0] > 0 and df.shape[1] > 1:
        return df

    # Fallback: handle "entire line quoted" CSVs.
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip("\r\n")
    except Exception as e:
        raise ValueError(f"Failed to read CSV header: {p} ({e})") from e

    if not first_line.strip().lstrip("\ufeff"):
        raise ValueError(f"CSV file is empty or missing header: {p}")

    header_core = first_line.lstrip("\ufeff").strip().strip('"')
    if "," not in header_core:
        # Nothing to infer; surface the original parse error if we have one.
        if first_error is not None:
            raise ValueError(f"Failed to parse CSV: {p} ({first_error})") from first_error
        raise ValueError(f"Failed to parse CSV: {p}")

    names = [h.strip() for h in header_core.split(",") if h.strip()]
    if not names:
        raise ValueError(f"Failed to infer column names from header: {p}")

    try:
        # quoting=3 corresponds to csv.QUOTE_NONE; using an int avoids importing csv.
        df2 = pd.read_csv(
            p,
            names=names,
            header=None,
            skiprows=1,
            engine="python",
            sep=",",
            quoting=3,
        )
    except Exception as e:
        if first_error is not None:
            raise ValueError(f"Failed to parse CSV (normal: {first_error}; fallback: {e}): {p}") from e
        raise ValueError(f"Failed to parse CSV: {p} ({e})") from e

    if df2.shape[0] == 0:
        raise ValueError(f"CSV file has no rows: {p}")

    # Strip surrounding quotes from string columns.
    out = df2.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.strip('"')

    return out


def stream_events_from_csv(path: str | Path, delay_seconds: float = 1.0) -> Iterator[dict]:
    """Stream events from a CSV row-by-row in original order.

    Args:
        path: CSV path.
        delay_seconds: Seconds to sleep between yielded events.

    Yields:
        Each row as a dict.
    """

    df = load_event_source(path)

    delay = float(delay_seconds)
    if delay < 0:
        delay = 0.0

    for row in df.to_dict(orient="records"):
        yield {str(k): v for k, v in row.items()}
        if delay > 0:
            time.sleep(delay)


def write_jsonl_record(record: dict, output_path: str | Path) -> None:
    """Append one JSONL record to a file.

    Creates the parent folder if needed.

    Args:
        record: JSON-serializable dict.
        output_path: Destination JSONL path.
    """

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        safe = _to_jsonable(record)
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")


def write_live_event(event: dict, output_path: str | Path) -> None:
    """Write one live event record (JSONL)."""

    write_jsonl_record(event, output_path)


def write_live_alert(alert: dict, output_path: str | Path) -> None:
    """Write one live alert record (JSONL)."""

    write_jsonl_record(alert, output_path)


def load_latest_jsonl_records(path: str | Path, limit: int = 50) -> list[dict]:
    """Load the last N JSONL records from a file.

    Args:
        path: JSONL path.
        limit: Max number of records to return.

    Returns:
        List of parsed JSON objects. Missing file returns an empty list.
    """

    p = Path(path)
    if not p.exists() or not p.is_file():
        return []

    n = int(limit)
    if n <= 0:
        return []

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    out: list[dict] = []
    for line in lines[-n:]:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
        else:
            out.append({"value": obj})
    return out


def clear_output_file(path: str | Path) -> None:
    """Delete an existing output file so a fresh simulation can start."""

    p = Path(path)
    try:
        if p.exists() and p.is_file():
            p.unlink()
    except Exception:
        # Best-effort cleanup.
        return


if __name__ == "__main__":
    # Small demo: load a CSV and stream a few events.
    default_csv = Path("Dataset/train/test/R2/email-train-data.csv")
    csv_path = default_csv if default_csv.exists() else None

    if csv_path is None:
        print("Demo CSV not found. Provide a path or place data under Dataset/.")
        raise SystemExit(0)

    print(f"Loading: {csv_path}")
    df0 = load_event_source(csv_path)
    print(f"Loaded rows={len(df0)} cols={list(df0.columns)}")

    print("Streaming 3 events (no delay):")
    started = datetime.now(timezone.utc)
    for i, ev in enumerate(stream_events_from_csv(csv_path, delay_seconds=0.0)):
        print(f"Event {i+1}: keys={list(ev.keys())}")
        if i >= 2:
            break
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"Done in {elapsed:.3f}s")
