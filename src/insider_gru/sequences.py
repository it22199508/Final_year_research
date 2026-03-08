"""Sequence construction utilities for insider threat modeling.

This module converts tabular behavioral features into fixed-length sequences for
sequential models (e.g., GRU, Transformer). It is designed to be deterministic,
explainable, and easy to debug for research workflows.

Typical usage:
- Start with a cleaned + feature-engineered DataFrame.
- Use `build_sequences` to create `(X, y)` for training.
- Use `build_latest_sequence_for_event` to construct a single most-recent
  sequence for real-time inference.

Dependencies: pandas, numpy, typing (standard library).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def summarize_label_distribution(y: np.ndarray) -> Dict[int, int]:
    """Return a simple label distribution dict for a 1D label array."""

    y_arr = np.asarray(y).reshape(-1)
    if y_arr.size == 0:
        return {}
    uniq, counts = np.unique(y_arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist(), strict=False)}


def format_label_distribution(dist: Dict[int, int]) -> str:
    """Format a distribution dict as a readable string with percentages."""

    if not dist:
        return "empty"

    total = float(sum(dist.values()))
    parts: List[str] = []
    for k in sorted(dist.keys()):
        v = int(dist[k])
        pct = (100.0 * float(v) / total) if total > 0 else 0.0
        parts.append(f"{int(k)}: {v} ({pct:.2f}%)")
    return ", ".join(parts)


def pad_or_trim_sequence(sequence: np.ndarray, seq_len: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad or trim a 2D feature sequence to a fixed length.

    Padding/trimming policy (deterministic):
    - If longer than `seq_len`, keep the most recent `seq_len` rows.
    - If shorter than `seq_len`, pad at the *beginning* so that the most recent
      events are aligned at the end of the sequence.

    This "pre-padding" is common in sequence models because it keeps the newest
    events in the same positions (right-aligned) across samples.

    Args:
        sequence: Array of shape (T, F) where T is timesteps and F is features.
        seq_len: Desired fixed sequence length.
        pad_value: Value used for padding rows.

    Returns:
        Array of shape (seq_len, F).

    Raises:
        ValueError: If seq_len <= 0 or sequence has incompatible shape.
    """

    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    seq = np.asarray(sequence)
    if seq.ndim == 1:
        seq = seq.reshape(-1, 1)
    if seq.ndim != 2:
        raise ValueError(f"sequence must be 2D (T, F); got shape {seq.shape}")

    t, f = seq.shape

    if t >= seq_len:
        return seq[-seq_len:, :]

    pad_rows = seq_len - t
    pad = np.full((pad_rows, f), pad_value, dtype=seq.dtype)
    return np.concatenate([pad, seq], axis=0)


def group_user_activity_windows(df: pd.DataFrame, user_col: str, time_col: str) -> Dict[str, pd.DataFrame]:
    """Group a DataFrame into per-user activity windows sorted by time.

    Args:
        df: Input DataFrame of events.
        user_col: Column that identifies the user/sender.
        time_col: Timestamp column used for ordering.

    Returns:
        Dict mapping user_id (as a string) to that user's events sorted by time.

    Raises:
        ValueError: If required columns are missing.
    """

    if user_col not in df.columns:
        raise ValueError(f"Missing user column: {user_col}")
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    # Do not mutate caller's df: we overwrite the time column with parsed datetimes.
    work = df.copy(deep=True)
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")

    # Stable sort for deterministic results.
    work = work.sort_values([user_col, time_col], kind="mergesort")

    groups: Dict[str, pd.DataFrame] = {}
    for user_value, g in work.groupby(user_col, sort=False):
        groups[str(user_value)] = g.copy(deep=True)

    return groups


def validate_sequence_inputs(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    user_col: str,
    time_col: str,
) -> None:
    """Validate required fields for sequence construction.

    Raises helpful errors rather than failing with cryptic pandas/numpy messages.

    Args:
        df: Input DataFrame.
        feature_cols: Feature columns to use.
        label_col: Label/target column name.
        user_col: User identifier column.
        time_col: Timestamp column.

    Raises:
        ValueError: If required columns are missing or feature_cols is empty.
    """

    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise ValueError("feature_cols must be a non-empty list of column names")

    required = set(feature_cols) | {label_col, user_col, time_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def _coerce_features_to_numpy(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Coerce selected feature columns into a numeric numpy array.

    Non-numeric values are converted to NaN and then filled with 0.0.
    """

    feats = df.loc[:, feature_cols].copy(deep=False)
    for c in feature_cols:
        feats[c] = pd.to_numeric(feats[c], errors="coerce")

    feats = feats.fillna(0.0)
    return feats.to_numpy(dtype=np.float32, copy=False)


def _coerce_labels_to_numpy(series: pd.Series) -> np.ndarray:
    """Coerce labels to a 1D numpy array.

    - If numeric/bool, coerces to int64.
    - Otherwise, uses deterministic categorical codes.

    Missing labels become -1.
    """

    s = series

    if pd.api.types.is_bool_dtype(s):
        out = s.fillna(False).astype(int).to_numpy(dtype=np.int64, copy=False)
        return out

    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce").fillna(-1).to_numpy(dtype=np.int64, copy=False)
        return out

    # Deterministic mapping based on sorted categories.
    cat = pd.Categorical(s.astype(str).fillna(""))
    codes = cat.codes.astype(np.int64, copy=False)
    return codes


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    user_col: str,
    time_col: str,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build rolling fixed-length sequences per user.

    For each user:
    - Sort events by time.
    - For each event i, create a sequence containing up to the last `seq_len`
      events ending at i (inclusive).
    - If fewer than `seq_len` events exist, pre-pad with zeros.
    - Use the label at event i as the target.

    This produces one training sample per labeled event.

    Args:
        df: Input DataFrame.
        feature_cols: List of feature columns to include.
        label_col: Label column; current event label is used as target.
        user_col: User identifier column.
        time_col: Timestamp column.
        seq_len: Fixed sequence length.

    Returns:
        (X, y)
        - X: shape (N, seq_len, num_features)
        - y: shape (N,)

    Raises:
        ValueError: If inputs are missing/invalid.
    """

    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    validate_sequence_inputs(df, feature_cols, label_col, user_col, time_col)

    grouped = group_user_activity_windows(df, user_col=user_col, time_col=time_col)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for _user, g in grouped.items():
        # Ensure time sort within group (group_user_activity_windows already sorts).
        g = g.sort_values(time_col, kind="mergesort")

        features = _coerce_features_to_numpy(g, feature_cols)
        labels = _coerce_labels_to_numpy(g[label_col])

        # Build a sample for each row with a valid label.
        for i in range(len(g)):
            yi = int(labels[i])
            if yi == -1:
                continue

            start = max(0, i - seq_len + 1)
            seq = features[start : i + 1, :]
            seq_fixed = pad_or_trim_sequence(seq, seq_len=seq_len, pad_value=0.0)

            X_list.append(seq_fixed)
            y_list.append(yi)

    if len(X_list) == 0:
        num_features = len(feature_cols)
        return (
            np.zeros((0, seq_len, num_features), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    y = np.asarray(y_list, dtype=np.int64)

    return X, y


def build_latest_sequence_for_event(
    df: pd.DataFrame,
    feature_cols: List[str],
    user_col: str,
    time_col: str,
    seq_len: int,
) -> np.ndarray:
    """Build the most recent fixed-length sequence for a single user.

    Intended for real-time inference where you have the latest events for a
    specific user and want a single model input.

    This function expects the input `df` to contain events for *one* user. If
    multiple users are present, it raises a clear error to avoid ambiguity.

    Args:
        df: DataFrame containing the user's most recent events.
        feature_cols: Feature columns to include.
        user_col: User identifier column.
        time_col: Timestamp column.
        seq_len: Fixed sequence length.

    Returns:
        Array of shape (1, seq_len, num_features).

    Raises:
        ValueError: If required fields are missing or df contains multiple users.
    """

    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    if user_col not in df.columns:
        raise ValueError(f"Missing user column: {user_col}")
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature column(s): {missing_feats}")

    users = pd.Series(df[user_col].astype(str).unique())
    if len(users) != 1:
        raise ValueError(
            f"build_latest_sequence_for_event expects exactly one user in df; got {len(users)} users"
        )

    # Do not mutate caller's df: we overwrite the time column with parsed datetimes.
    work = df.copy(deep=True)
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.sort_values(time_col, kind="mergesort")

    features = _coerce_features_to_numpy(work, feature_cols)
    latest_seq = features[-seq_len:, :]
    latest_fixed = pad_or_trim_sequence(latest_seq, seq_len=seq_len, pad_value=0.0)

    return latest_fixed[np.newaxis, :, :].astype(np.float32, copy=False)
