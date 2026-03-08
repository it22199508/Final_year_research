"""Preprocessing utilities for insider threat event data.

This module provides reusable, privacy-aware cleaning steps for raw CSV event logs
(e.g., email activity and login/device activity) before feature engineering and
model training.

Why hashing?
------------
Event data often contains personally identifiable information (PII) such as email
addresses, usernames, device names, or IPs. Hashing these fields (SHA-256) helps
reduce privacy exposure while preserving *joinability* (the same identifier always
maps to the same hash) for analytics.

Notes
-----
- Functions return copies where mutation might be surprising.
- Timestamp parsing uses `errors="coerce"` to avoid crashing on malformed inputs.
- Missing-value filling is deterministic and intentionally simple.

Dependencies: pandas + standard library (hashlib, pathlib, typing).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_event_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with clear, safe errors.

    Args:
        path: File path to a CSV file.

    Returns:
        A non-empty pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or cannot be parsed as a CSV.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Event data file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Expected a file path, got: {p}")

    # Fast empty-file check (still let pandas raise if it finds no columns).
    try:
        if p.stat().st_size == 0:
            raise ValueError(f"Event data file is empty: {p}")
    except OSError:
        # If stat fails, fall back to pandas read.
        pass

    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Event data file is empty or has no columns: {p}") from e
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {p}. Details: {e}") from e

    if df.shape[0] == 0:
        raise ValueError(f"Event data file contains no rows: {p}")

    return df


def _slugify_column_name(name: str) -> str:
    """Normalize a single column name into lowercase underscore format."""

    s = str(name).strip().lower()
    # Replace common separators with underscores
    for ch in (" ", "-", ".", "/", "\\", ":", "\t", "\n", "\r"):
        s = s.replace(ch, "_")
    # Collapse repeated underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores.

    Behavior:
        - Strips whitespace
        - Converts to lowercase
        - Replaces spaces and common separators with underscores
        - Ensures uniqueness by suffixing duplicates with "__2", "__3", ...

    Args:
        df: Input DataFrame.

    Returns:
        A copy of the DataFrame with normalized column names.
    """

    out = df.copy(deep=False)

    normalized: list[str] = [_slugify_column_name(c) for c in out.columns]

    # Handle duplicates safely by adding deterministic suffixes.
    seen: dict[str, int] = {}
    unique: list[str] = []
    for col in normalized:
        base = col or "col"
        if base not in seen:
            seen[base] = 1
            unique.append(base)
            continue
        seen[base] += 1
        unique.append(f"{base}__{seen[base]}")

    out.columns = unique
    return out


def parse_timestamps(df: pd.DataFrame, timestamp_columns: list[str]) -> pd.DataFrame:
    """Parse specified columns as datetimes.

    Invalid values are coerced to NaT. The number of rows and columns is preserved.

    Args:
        df: Input DataFrame.
        timestamp_columns: Column names to convert using pandas datetime parsing.

    Returns:
        A copy of the DataFrame with parsed datetime columns.

    Raises:
        ValueError: If any requested timestamp column is missing.
    """

    # Use a deep copy because we overwrite existing column values.
    out = df.copy(deep=True)

    missing = [c for c in timestamp_columns if c not in out.columns]
    if missing:
        raise ValueError(f"Missing timestamp column(s): {missing}")

    for col in timestamp_columns:
        out[col] = pd.to_datetime(out[col], errors="coerce", utc=False)

    return out


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with deterministic defaults.

    Rules:
        - Numeric columns: fill NaN with 0
        - Object/string columns: fill NaN with empty string

    Args:
        df: Input DataFrame.

    Returns:
        A copy of the DataFrame with missing values filled.
    """

    # Use a deep copy because we overwrite missing values in existing columns.
    out = df.copy(deep=True)

    numeric_cols = out.select_dtypes(include=["number"]).columns
    object_cols = out.select_dtypes(include=["object", "string"]).columns

    if len(numeric_cols) > 0:
        out.loc[:, numeric_cols] = out.loc[:, numeric_cols].fillna(0)

    if len(object_cols) > 0:
        out.loc[:, object_cols] = out.loc[:, object_cols].fillna("")

    return out


def hash_value(value: str) -> str:
    """Hash a value using SHA-256 and return a hex digest.

    Args:
        value: Input string.

    Returns:
        SHA-256 hex digest.
    """

    data = value.encode("utf-8", errors="ignore")
    return hashlib.sha256(data).hexdigest()


def _as_string_or_empty(x: object) -> str:
    """Convert values to string safely; keep missing values as empty string."""

    # pandas NA/NaN handling
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def hash_pii_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Hash selected PII columns (e.g., sender, recipient, user, device, ip).

    This function returns a copy and does not mutate the input DataFrame.

    Args:
        df: Input DataFrame.
        columns: List of column names to hash.

    Returns:
        A copy of the DataFrame with the specified columns hashed.

    Raises:
        ValueError: If any requested PII column is missing.
    """

    # Use a deep copy because we overwrite selected PII column values.
    out = df.copy(deep=True)

    missing = [c for c in columns if c not in out.columns]
    if missing:
        raise ValueError(f"Missing PII column(s): {missing}")

    for col in columns:
        out[col] = out[col].map(lambda v: hash_value(_as_string_or_empty(v)))

    return out


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that required columns exist.

    Args:
        df: Input DataFrame.
        required_columns: Column names that must be present.

    Raises:
        ValueError: If any required columns are missing.
    """

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def clean_event_data(
    df: pd.DataFrame,
    timestamp_columns: list[str] | None = None,
    pii_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Clean event data for downstream feature engineering.

    Steps:
        1) Normalize column names
        2) Parse timestamps (optional)
        3) Fill missing values
        4) Hash PII columns (optional)

    Args:
        df: Raw input DataFrame.
        timestamp_columns: Column names to parse as datetimes. If None, no parsing.
        pii_columns: Column names to hash with SHA-256. If None, no hashing.

    Returns:
        A cleaned DataFrame ready for feature engineering.
    """

    out = normalize_columns(df)

    if timestamp_columns:
        # Normalize requested timestamp column names to match normalized df columns.
        ts_cols = [_slugify_column_name(c) for c in timestamp_columns]
        out = parse_timestamps(out, ts_cols)

    out = fill_missing_values(out)

    if pii_columns:
        # Normalize requested PII column names to match normalized df columns.
        pii_cols = [_slugify_column_name(c) for c in pii_columns]
        out = hash_pii_columns(out, pii_cols)

    return out


def _guess_pii_columns(columns: Iterable[str]) -> list[str]:
    """Heuristic helper for the demo block: guess likely PII columns."""

    candidates = {
        "from",
        "to",
        "sender",
        "recipient",
        "user",
        "username",
        "pc",
        "device",
        "ip",
        "src_ip",
        "dst_ip",
        "email",
    }
    cols = [str(c) for c in columns]
    return [c for c in cols if c in candidates]


if __name__ == "__main__":
    # Demo usage:
    #   python -m insider_gru.preprocessing <path/to/events.csv>
    #
    # The demo:
    # - loads a CSV
    # - normalizes columns
    # - attempts to parse a common timestamp column
    # - hashes likely PII columns
    #
    # Hashing is used here for privacy-preserving preprocessing: it helps reduce exposure
    # of direct identifiers (emails/usernames/devices) while keeping consistent IDs for
    # grouping and analysis.

    if len(sys.argv) < 2:
        print("Usage: python -m insider_gru.preprocessing <path/to/events.csv>")
        raise SystemExit(2)

    csv_path = Path(sys.argv[1])
    df0 = load_event_data(csv_path)
    df0 = normalize_columns(df0)

    # Common timestamp column names in this repo/dataset.
    timestamp_candidates = [c for c in ["date", "timestamp", "time"] if c in df0.columns]

    cleaned = clean_event_data(
        df0,
        timestamp_columns=timestamp_candidates or None,
        pii_columns=_guess_pii_columns(df0.columns),
    )

    print(f"Loaded rows: {len(df0):,}")
    print(f"Columns: {list(cleaned.columns)}")
    print(cleaned.head(5).to_string(index=False))
