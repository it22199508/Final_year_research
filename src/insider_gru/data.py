from __future__ import annotations

import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from insider_gru import features as feat
from insider_gru import preprocessing as preproc
from insider_gru import sequences as seq


"""Data preparation for model training.

This module is the training-data layer of the project.

Recommended (new) API
---------------------
Use these functions for simplified end-to-end preparation:
- prepare_email_training_data
- prepare_login_training_data
- select_numeric_feature_columns
- train_test_split_sequences
- summarize_dataset

Legacy API
----------
The bottom half of this file contains legacy helpers (e.g., CERT CSV parsing,
classic sklearn preprocessors, and an older sequence window builder) that are
still imported by existing scripts/apps in this repo.
"""


DatasetSplit = Literal["train", "test"]

# Canonical (research-friendly) schema for device log CSVs when headerless.
# Note: the rest of this repo historically uses (id,date,...) so the loader
# also creates id/date aliases for compatibility.
DEVICE_LOG_COLUMNS = ["event_id", "timestamp", "user", "pc", "activity"]


def resolve_train_test_split_dirs(dataset_root: str | Path) -> tuple[Path, Path]:
    """Resolve the train/test split directories for this project.

    Supported layouts:
      1) Nested test split (your real structure):
         Dataset/train/R1..R4
         Dataset/train/test/R1..R4
      2) Legacy flat test split:
         Dataset/train/R1..R4
         Dataset/test/R1..R4

    Returns:
        (train_dir, test_dir)
    """

    root = Path(dataset_root)
    train_dir = root / "train"

    # Prefer the nested test split if it exists.
    nested_test_dir = train_dir / "test"
    if nested_test_dir.exists() and nested_test_dir.is_dir():
        test_dir = nested_test_dir
    else:
        test_dir = root / "test"

    return train_dir, test_dir


def resolve_split_dir(dataset_root: str | Path, split: DatasetSplit) -> Path:
    """Resolve a single split directory (train or test) from dataset_root."""

    train_dir, test_dir = resolve_train_test_split_dirs(dataset_root)
    return train_dir if str(split) == "train" else test_dir


def discover_group_folders(base_path: str | Path) -> list[Path]:
    """Discover R-group folders under a base path.

        This matches either dataset structure:
                Dataset/train/R1..R4
                Dataset/train/test/R1..R4   (preferred)
            or
                Dataset/test/R1..R4         (legacy)

    Args:
        base_path: Path to a split directory such as Dataset/train or Dataset/train/test.

    Returns:
        Sorted list of group folder Paths (e.g., [Dataset/test/R1, Dataset/test/R2, ...]).
    """

    base = Path(base_path)
    if not base.exists() or not base.is_dir():
        return []

    groups: list[Path] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        name = child.name.strip()
        if name.lower().startswith("r") and name[1:].isdigit():
            groups.append(child)

    def _sort_key(p: Path) -> tuple[int, str]:
        name = p.name.strip()
        try:
            return (int(name[1:]), name)
        except Exception:
            return (10**9, name)

    return sorted(groups, key=_sort_key)


def load_combined_event_data(
    base_path: str | Path,
    event_type: str,
    *,
    selected_group: str | None = None,
) -> pd.DataFrame:
    """Load and combine all matching event CSVs under a split directory.

    This is a thin convenience wrapper around the split-aware scanning logic.

    Args:
        base_path: Path to a split directory such as Dataset/train or Dataset/train/test.
        event_type: "device" (or "login"), "email", or "http".
        selected_group: Optional group like "R1".

    Returns:
        Combined DataFrame across all matched files.
    """

    base = Path(base_path)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Base path not found: {base}")

    groups = discover_group_folders(base)
    if not groups:
        raise FileNotFoundError(f"No R-folders found under: {base}")

    if selected_group is not None:
        g = str(selected_group).strip()
        groups = [p for p in groups if p.name == g]
        if not groups:
            raise ValueError(f"Selected group {g!r} not found under {base}")

    frames: list[pd.DataFrame] = []
    for group_dir in groups:
        for p in find_event_csv_files(group_dir, event_type):
            df = _read_csv(p, event_type=event_type)
            if df.empty:
                continue
            df["__group__"] = group_dir.name
            df["__source_file__"] = str(p).replace("\\", "/")
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No CSVs found for event_type={event_type!r} under {base} "
            f"(groups={[p.name for p in groups]})."
        )

    return pd.concat(frames, ignore_index=True)


def load_train_test_event_data(
    train_base: str | Path,
    test_base: str | Path,
    event_type: str,
    *,
    allow_fallback_if_no_test: bool = False,
    fallback_test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test dataframes from explicit split directories.

    If the test split is missing or contains no matched CSVs, an optional fallback
    can create a pseudo-test set by randomly splitting the *train* data.

    IMPORTANT: This fallback is disabled by default to avoid accidental leakage.

    Args:
        train_base: Dataset/train
        test_base: Dataset/train/test (preferred) or Dataset/test (legacy)
        event_type: "device" (or "login"), "email", or "http".
        allow_fallback_if_no_test: If True, and explicit test data is unavailable,
            returns a random split of the train dataframe.
        fallback_test_size: Test fraction for fallback.
        seed: Random seed for fallback.

    Returns:
        (train_df, test_df)
    """

    train_df = load_combined_event_data(train_base, event_type)
    try:
        test_df = load_combined_event_data(test_base, event_type)
    except Exception:
        if not allow_fallback_if_no_test:
            raise
        train_df, test_df = train_test_split(
            train_df,
            test_size=float(fallback_test_size),
            random_state=int(seed),
            shuffle=True,
        )

    return train_df, test_df


def discover_dataset_groups(dataset_root: str | Path, split: DatasetSplit) -> list[str]:
    """Discover available R-groups under a split directory.

    Expected dataset layout (preferred):
        Dataset/train/R1 .. Dataset/train/R4
        Dataset/train/test/R1 .. Dataset/train/test/R4

    Legacy dataset layout (also supported):
        Dataset/train/R1 .. Dataset/train/R4
        Dataset/test/R1 .. Dataset/test/R4

    Args:
        dataset_root: Path to the Dataset directory.
        split: "train" or "test".

    Returns:
        List of group folder names (e.g., ["R1", "R2", ...]) sorted by numeric suffix.
    """

    split_dir = resolve_split_dir(dataset_root, split)
    if not split_dir.exists() or not split_dir.is_dir():
        return []

    groups: list[str] = []
    for child in split_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.strip()
        if name.lower().startswith("r") and name[1:].isdigit():
            groups.append(name)

    def _sort_key(g: str) -> tuple[int, str]:
        try:
            return (int(g[1:]), g)
        except Exception:
            return (10**9, g)

    return sorted(groups, key=_sort_key)


def find_event_csv_files(folder_path: str | Path, event_type: str) -> list[Path]:
    """Find event CSV files for a given event_type inside a group folder.

    This loader is conservative: it prefers files whose names include the event type
    token (device/email/http) to avoid mixing modalities.

    Args:
        folder_path: Path to a group folder (e.g., Dataset/train/R2).
        event_type: "email", "device", "http" ("login" is treated as "device").

    Returns:
        Sorted list of CSV Paths.
    """

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []

    key = str(event_type).strip().lower()
    if key == "login":
        key = "device"

    # Don't rely on glob case-sensitivity across platforms.
    csv_paths = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    ]

    def _match(p: Path) -> bool:
        name = p.name.lower()
        if key == "email":
            return "email" in name
        if key == "device":
            # Device logs may be named with either "device" or legacy "logon" tokens.
            return ("device" in name) or ("logon" in name)
        if key == "http":
            return "http" in name
        return key in name

    return sorted([p for p in csv_paths if _match(p)])


def load_combined_event_logs_from_split(
    dataset_root: str | Path,
    split: DatasetSplit,
    event_type: str,
    *,
    selected_group: str | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Load and combine all matching event logs for a split.

        The split is *strictly enforced* by folder location.

        Preferred layout:
            - training data: Dataset/train/R*
            - evaluation data: Dataset/train/test/R*

        Legacy layout (also supported):
            - training data: Dataset/train/R*
            - evaluation data: Dataset/test/R*

    Args:
        dataset_root: Path to Dataset.
        split: "train" or "test".
        event_type: "email", "device", "http".
        selected_group: Optional group like "R1". If None, loads all groups.

    Returns:
        (df, paths)
        df includes extra metadata columns:
          - __split__ (train/test)
          - __group__ (R1/R2/...)
          - __source_file__ (path string)
    """

    split_dir = resolve_split_dir(dataset_root, split)
    if not split_dir.exists() or not split_dir.is_dir():
        train_dir, test_dir = resolve_train_test_split_dirs(dataset_root)
        raise FileNotFoundError(
            "Split directory not found: "
            f"{split_dir}. Resolved train={train_dir}, test={test_dir}"
        )

    available_groups = discover_dataset_groups(dataset_root, split)
    if not available_groups:
        raise FileNotFoundError(f"No R-folders found under: {split_dir}")

    if selected_group is None:
        groups = available_groups
    else:
        g = str(selected_group).strip()
        if g not in available_groups:
            raise ValueError(f"Selected group {g!r} not found under {split_dir}")
        groups = [g]

    frames: list[pd.DataFrame] = []
    used_paths: list[Path] = []

    for g in groups:
        group_dir = split_dir / g
        paths = find_event_csv_files(group_dir, event_type)
        for p in paths:
            try:
                df = _read_csv(p, event_type=event_type)
            except Exception as e:
                print(f"[data] Skipping invalid CSV: {p} ({type(e).__name__}: {e})")
                continue

            if df.empty:
                print(f"[data] Skipping empty CSV: {p}")
                continue

            df["__split__"] = str(split)
            df["__group__"] = g
            df["__source_file__"] = str(p).replace("\\", "/")
            frames.append(df)
            used_paths.append(p)

    if not frames:
        raise FileNotFoundError(
            f"No CSVs found for event_type={event_type!r} under {split_dir} "
            f"(groups={groups})."
        )

    df_all = pd.concat(frames, ignore_index=True)
    return df_all, used_paths


def load_train_test_event_logs(
    dataset_root: str | Path,
    event_type: str,
    *,
    selected_group: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path], list[Path]]:
    """Convenience: load train and test splits for an event_type."""

    train_df, train_paths = load_combined_event_logs_from_split(
        dataset_root,
        "train",
        event_type,
        selected_group=selected_group,
    )
    test_df, test_paths = load_combined_event_logs_from_split(
        dataset_root,
        "test",
        event_type,
        selected_group=selected_group,
    )
    return train_df, test_df, train_paths, test_paths


def _norm_col(name: str) -> str:
    """Normalize a column name in the same style as preprocessing.normalize_columns."""

    s = str(name).strip().lower()
    for ch in (" ", "-", ".", "/", "\\", ":", "\t", "\n", "\r"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def select_numeric_feature_columns(df: pd.DataFrame, exclude_cols: Optional[Sequence[str]] = None) -> list[str]:
    """Select numeric feature columns suitable for modeling.

    This helper is intentionally conservative and explainable:
    - It only selects numeric/bool dtypes.
    - It excludes any columns explicitly passed in `exclude_cols`.

    Args:
        df: Input DataFrame.
        exclude_cols: Column names to exclude (labels, IDs, timestamps, etc.).

    Returns:
        List of numeric feature column names, preserving DataFrame column order.
    """

    exclude = set(exclude_cols or [])
    numeric_cols = list(df.select_dtypes(include=["number", "bool"]).columns)
    return [c for c in numeric_cols if c not in exclude]


def train_test_split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequence datasets into train/test using sklearn.

    Args:
        X: Feature sequences of shape (N, seq_len, num_features).
        y: Labels of shape (N,).
        test_size: Fraction to use for test split.
        random_state: Random seed.
        stratify: If True, attempts stratified splitting when feasible.

    Returns:
        X_train, X_test, y_train, y_test
    """

    y_arr = np.asarray(y)
    stratify_vec = None
    if stratify:
        unique, counts = np.unique(y_arr, return_counts=True)
        # Stratification requires at least 2 classes and at least 2 samples per class.
        if unique.size >= 2 and int(counts.min()) >= 2:
            stratify_vec = y_arr

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_arr,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=stratify_vec,
    )

    return X_train, X_test, y_train, y_test


def summarize_dataset(X: np.ndarray, y: np.ndarray, feature_cols: Sequence[str]) -> dict[str, Any]:
    """Summarize a prepared sequence dataset.

    Args:
        X: Feature tensor (N, seq_len, num_features).
        y: Labels (N,).
        feature_cols: List of feature column names used in X.

    Returns:
        Dict containing sample count, sequence length, feature count, and label distribution.
    """

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    n = int(X_arr.shape[0]) if X_arr.ndim >= 1 else 0
    seq_len = int(X_arr.shape[1]) if X_arr.ndim >= 2 else 0
    n_feat = int(X_arr.shape[2]) if X_arr.ndim >= 3 else int(len(feature_cols))

    uniq, counts = np.unique(y_arr, return_counts=True) if y_arr.size else (np.array([]), np.array([]))
    label_dist = {str(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist(), strict=False)}

    return {
        "num_samples": n,
        "sequence_length": seq_len,
        "num_features": n_feat,
        "feature_cols": list(feature_cols),
        "label_distribution": label_dist,
    }


def _canonicalize_email_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical email identity columns exist for feature engineering."""

    out = df.copy(deep=True)
    if "sender" not in out.columns and "from" in out.columns:
        out["sender"] = out["from"]
    if "recipient" not in out.columns and "to" in out.columns:
        out["recipient"] = out["to"]
    return out


def _canonicalize_login_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical login identity columns exist for feature engineering."""

    out = df.copy(deep=True)
    if "user" not in out.columns and "username" in out.columns:
        out["user"] = out["username"]
    return out


def _load_single_csv_like_legacy(path: Path) -> pd.DataFrame:
    """Load one CSV using the legacy CERT-tolerant parser if possible."""

    # If the path doesn't exist, let the caller raise a helpful error.
    return _read_csv(path)


TrainingDataSource = Union[str, Path, pd.DataFrame]


def _load_training_dataframe(data_path: TrainingDataSource) -> pd.DataFrame:
    """Load training data from a file, directory, or glob.

    For compatibility with this repo's CERT-style dataset files, this loader
    supports:
    - single CSV file
    - directory containing CSVs
    - glob pattern

    Args:
        data_path: File path, directory, or glob.

    Returns:
        Concatenated DataFrame.
    """

    if isinstance(data_path, pd.DataFrame):
        return data_path.copy(deep=True)

    p = Path(data_path)

    if p.exists() and p.is_file():
        return _load_single_csv_like_legacy(p)

    if p.exists() and p.is_dir():
        paths = sorted(p.glob("*.csv"))
        return load_event_csvs(paths)

    # Treat as glob (relative to cwd) when the path doesn't exist.
    paths = sorted(Path(".").glob(str(data_path)))
    if paths:
        return load_event_csvs(paths)

    raise FileNotFoundError(f"No files found for data_path: {data_path}")


def prepare_email_training_data(
    data_path: TrainingDataSource,
    label_col: str,
    user_col: str,
    time_col: str,
    feature_cols: Optional[list[str]] = None,
    seq_len: int = 20,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Prepare email training data end-to-end: load → clean → features → sequences.

    Args:
        data_path: CSV path, directory of CSVs, or glob pattern.
        label_col: Label column name in the raw data.
        user_col: Column name identifying the user/sender for grouping.
        time_col: Timestamp column name.
        feature_cols: Optional explicit list of feature columns. If None, numeric
            features will be selected automatically.
        seq_len: Fixed sequence length.

    Returns:
        (X, y, feature_cols, processed_df)
        - X: (N, seq_len, num_features)
        - y: (N,)
        - feature_cols: columns used in X
        - processed_df: engineered DataFrame used for sequence construction
    """

    raw_df = _load_training_dataframe(data_path)

    if debug:
        # Raw label distribution before any cleaning/preprocessing.
        raw_label_series = None
        if label_col in raw_df.columns:
            raw_label_series = raw_df[label_col]
        else:
            # Best-effort fallback: if caller already passed normalized names.
            ln = _norm_col(label_col)
            if ln in raw_df.columns:
                raw_label_series = raw_df[ln]
        if raw_label_series is not None:
            print(
                f"[prepare_email_training_data] RAW labels before preprocessing: col={label_col!r} "
                f"rows={len(raw_df)} dtype={raw_label_series.dtype}"
            )
            print(raw_label_series.value_counts(dropna=False).to_string())
        else:
            print(
                f"[prepare_email_training_data] RAW labels before preprocessing: col={label_col!r} "
                "(column not found in raw_df)"
            )

    # Clean (normalizes column names internally).
    cleaned = preproc.clean_event_data(raw_df, timestamp_columns=[time_col])
    cleaned = _canonicalize_email_id_columns(cleaned)

    label_norm = _norm_col(label_col)
    user_norm = _norm_col(user_col)
    time_norm = _norm_col(time_col)

    if label_norm not in cleaned.columns:
        raise ValueError(f"Missing label column after cleaning: {label_col!r} (normalized: {label_norm!r})")
    if user_norm not in cleaned.columns:
        raise ValueError(f"Missing user column after cleaning: {user_col!r} (normalized: {user_norm!r})")
    if time_norm not in cleaned.columns:
        raise ValueError(f"Missing time column after cleaning: {time_col!r} (normalized: {time_norm!r})")

    if debug:
        print(
            f"[prepare_email_training_data] labels after preprocessing/cleaning (pre-features): "
            f"col={label_norm!r} rows={len(cleaned)} dtype={cleaned[label_norm].dtype}"
        )
        print(cleaned[label_norm].value_counts(dropna=False).to_string())

    engineered = feat.engineer_email_features(
        cleaned,
        timestamp_col=time_norm,
        recent_window_hours=feat.DEFAULT_RECENT_WINDOW_HOURS,
    )

    if debug:
        print(
            f"[prepare_email_training_data] label_col={label_col!r} normalized={label_norm!r} "
            f"rows={len(engineered)} dtype={engineered[label_norm].dtype}"
        )
        vc = engineered[label_norm].value_counts(dropna=False)
        print("[prepare_email_training_data] labels before normalization:")
        print(vc.to_string())

    # Labels from mapping files are often sparse (only positives listed). For training we treat
    # missing labels as negative (0), matching legacy normalize_binary_labels behavior.
    engineered = engineered.copy(deep=True)
    engineered[label_norm] = normalize_binary_labels(engineered[label_norm])

    if debug:
        vc2 = engineered[label_norm].value_counts(dropna=False)
        print("[prepare_email_training_data] labels after normalization (pre-sequences):")
        print(vc2.to_string())

    if feature_cols is None:
        exclude: list[str] = [label_norm, user_norm, time_norm]
        if "id" in engineered.columns:
            exclude.append("id")
        feature_cols = select_numeric_feature_columns(engineered, exclude_cols=exclude)
    else:
        missing = [c for c in feature_cols if c not in engineered.columns]
        if missing:
            raise ValueError(f"Requested feature_cols missing after engineering: {missing}")

    # Build sequences using the standardized sequences module.
    X, y = seq.build_sequences(
        engineered,
        feature_cols=feature_cols,
        label_col=label_norm,
        user_col=user_norm,
        time_col=time_norm,
        seq_len=int(seq_len),
    )

    if debug:
        uniq, counts = np.unique(y, return_counts=True) if y.size else (np.array([]), np.array([]))
        dist = {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist(), strict=False)}
        print(f"[prepare_email_training_data] labels after sequence building: {dist}")

    return X, y, list(feature_cols), engineered


def prepare_login_training_data(
    data_path: TrainingDataSource,
    label_col: str,
    user_col: str,
    time_col: str,
    feature_cols: Optional[list[str]] = None,
    seq_len: int = 20,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Prepare login/device training data end-to-end: load → clean → features → sequences."""

    raw_df = _load_training_dataframe(data_path)

    if debug:
        raw_label_series = None
        if label_col in raw_df.columns:
            raw_label_series = raw_df[label_col]
        else:
            ln = _norm_col(label_col)
            if ln in raw_df.columns:
                raw_label_series = raw_df[ln]
        if raw_label_series is not None:
            print(
                f"[prepare_login_training_data] RAW labels before preprocessing: col={label_col!r} "
                f"rows={len(raw_df)} dtype={raw_label_series.dtype}"
            )
            print(raw_label_series.value_counts(dropna=False).to_string())
        else:
            print(
                f"[prepare_login_training_data] RAW labels before preprocessing: col={label_col!r} "
                "(column not found in raw_df)"
            )

    cleaned = preproc.clean_event_data(raw_df, timestamp_columns=[time_col])
    cleaned = _canonicalize_login_id_columns(cleaned)

    label_norm = _norm_col(label_col)
    user_norm = _norm_col(user_col)
    time_norm = _norm_col(time_col)

    if label_norm not in cleaned.columns:
        raise ValueError(f"Missing label column after cleaning: {label_col!r} (normalized: {label_norm!r})")
    if user_norm not in cleaned.columns:
        raise ValueError(f"Missing user column after cleaning: {user_col!r} (normalized: {user_norm!r})")
    if time_norm not in cleaned.columns:
        raise ValueError(f"Missing time column after cleaning: {time_col!r} (normalized: {time_norm!r})")

    if debug:
        print(
            f"[prepare_login_training_data] labels after preprocessing/cleaning (pre-features): "
            f"col={label_norm!r} rows={len(cleaned)} dtype={cleaned[label_norm].dtype}"
        )
        print(cleaned[label_norm].value_counts(dropna=False).to_string())

    engineered = feat.engineer_login_features(
        cleaned,
        timestamp_col=time_norm,
        recent_window_hours=feat.DEFAULT_RECENT_WINDOW_HOURS,
    )

    if debug:
        print(
            f"[prepare_login_training_data] label_col={label_col!r} normalized={label_norm!r} "
            f"rows={len(engineered)} dtype={engineered[label_norm].dtype}"
        )
        vc = engineered[label_norm].value_counts(dropna=False)
        print("[prepare_login_training_data] labels before normalization:")
        print(vc.to_string())

    engineered = engineered.copy(deep=True)
    engineered[label_norm] = normalize_binary_labels(engineered[label_norm])

    if debug:
        vc2 = engineered[label_norm].value_counts(dropna=False)
        print("[prepare_login_training_data] labels after normalization (pre-sequences):")
        print(vc2.to_string())

    if feature_cols is None:
        exclude: list[str] = [label_norm, user_norm, time_norm]
        if "id" in engineered.columns:
            exclude.append("id")
        feature_cols = select_numeric_feature_columns(engineered, exclude_cols=exclude)
    else:
        missing = [c for c in feature_cols if c not in engineered.columns]
        if missing:
            raise ValueError(f"Requested feature_cols missing after engineering: {missing}")

    X, y = seq.build_sequences(
        engineered,
        feature_cols=feature_cols,
        label_col=label_norm,
        user_col=user_norm,
        time_col=time_norm,
        seq_len=int(seq_len),
    )

    if debug:
        uniq, counts = np.unique(y, return_counts=True) if y.size else (np.array([]), np.array([]))
        dist = {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist(), strict=False)}
        print(f"[prepare_login_training_data] labels after sequence building: {dist}")

    return X, y, list(feature_cols), engineered


@dataclass(frozen=True)
class PreparedData:
    df: pd.DataFrame
    group_col: str
    timestamp_col: str
    id_col: str
    feature_cols: list[str]


def _read_csv(path: Path, event_type: str | None = None) -> pd.DataFrame:
    # Dataset files are inconsistent:
    # - some include a header row (sometimes quoted)
    # - some have NO header row (first line is data)
    # Also, many files wrap each line in quotes.

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip("\r\n")

    first_line = first_line.lstrip("\ufeff").strip()

    def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].astype(str).str.strip('"')
        return out

    def _ensure_device_schema(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Ensure required columns exist
        for c in DEVICE_LOG_COLUMNS:
            if c not in out.columns:
                out[c] = ""
        # Keep only canonical order + any extras
        ordered = [c for c in DEVICE_LOG_COLUMNS if c in out.columns]
        extra = [c for c in out.columns if c not in ordered]
        out = out[ordered + extra].copy()
        # Compatibility aliases for the rest of the pipeline
        if "id" not in out.columns:
            out["id"] = out["event_id"]
        if "date" not in out.columns:
            out["date"] = out["timestamp"]
        return out

    def _is_header(line: str) -> bool:
        # Heuristic: header contains commas and alphabetic column names, and does not look like an id record.
        core = line.strip().strip('"')
        if "," not in core:
            return False
        if "{" in core or "}" in core:
            return False
        # must contain at least one alpha character
        return any(ch.isalpha() for ch in core)

    def _infer_names_from_row(line: str) -> list[str]:
        core = line.strip().strip('"')
        parts = core.split(",")
        n = len(parts)
        if n == 7:
            return ["id", "date", "to", "from", "size", "attachments", "content"]
        if n == 5:
            # Disambiguate http vs device by inspecting the 5th field
            last = parts[-1].strip().lower()
            if last.startswith("http://") or last.startswith("https://"):
                return ["id", "date", "user", "pc", "url"]
            return ["id", "date", "user", "pc", "activity"]
        raise ValueError(f"Cannot infer schema for {path} (columns={n}).")

    # Event-specific parsing (used by fixed train/test split loaders)
    if event_type is not None and str(event_type).strip().lower() in {"device", "login"}:
        # Always read device logs as headerless (header=None) and assign canonical columns.
        skip = 1 if _is_header(first_line) else 0
        df = pd.read_csv(
            path,
            header=None,
            names=DEVICE_LOG_COLUMNS,
            skiprows=skip,
            engine="python",
            sep=",",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
            dtype=str,
        )
        # If extra columns exist, keep only the first 5.
        if df.shape[1] > len(DEVICE_LOG_COLUMNS):
            df = df.iloc[:, : len(DEVICE_LOG_COLUMNS)].copy()
            df.columns = DEVICE_LOG_COLUMNS
        return _ensure_device_schema(_postprocess(df))

    if _is_header(first_line):
        header = first_line.strip().strip('"')
        names = [h.strip() for h in header.split(",") if h.strip()]
        df = pd.read_csv(
            path,
            names=names,
            header=None,
            skiprows=1,
            engine="python",
            sep=",",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
        )
        return _postprocess(df)

    # No header: infer column names from first row
    if event_type is not None:
        key = str(event_type).strip().lower()
        if key == "login":
            key = "device"
        if key == "email":
            names = ["id", "date", "to", "from", "size", "attachments", "content"]
        elif key == "http":
            names = ["id", "date", "user", "pc", "url"]
        else:
            names = _infer_names_from_row(first_line)
    else:
        names = _infer_names_from_row(first_line)
    df = pd.read_csv(
        path,
        names=names,
        header=None,
        engine="python",
        sep=",",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )
    return _postprocess(df)


def load_event_csvs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = _read_csv(p)
        df["__source_file__"] = str(p).replace("\\", "/")
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No CSV files found for the provided pattern(s).")
    return pd.concat(frames, ignore_index=True)


def ensure_datetime(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found. Columns: {list(df.columns)}")
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    if out[timestamp_col].isna().any():
        # keep rows with valid timestamps
        out = out.dropna(subset=[timestamp_col]).copy()
    return out


def pick_group_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: if this is email data, use 'from' if exists, else single stream
    return "__all__"


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = out[timestamp_col]
    out["hour"] = ts.dt.hour.astype("int16")
    out["dayofweek"] = ts.dt.dayofweek.astype("int16")
    out["day"] = ts.dt.day.astype("int16")
    out["month"] = ts.dt.month.astype("int16")
    return out


_domain_re = re.compile(r"^[a-zA-Z]+://([^/]+)/?.*$")


def extract_domain(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    m = _domain_re.match(url.strip())
    if not m:
        return ""
    return m.group(1).lower()


def add_basic_text_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Email content length if present
    if "content" in out.columns:
        out["content_len"] = out["content"].astype(str).str.len().astype("int32")

    # URLs: domain and length
    if "url" in out.columns:
        out["url_domain"] = out["url"].astype(str).map(extract_domain)
        out["url_len"] = out["url"].astype(str).str.len().astype("int32")

    return out


def attach_labels(
    df: pd.DataFrame,
    id_col: str,
    label_col: str | None,
    label_mapping_csv: Path | None,
) -> pd.DataFrame:
    out = df.copy()

    if label_col and label_col in out.columns:
        return out

    if label_mapping_csv is None:
        return out

    if not label_mapping_csv.exists():
        raise FileNotFoundError(f"Label mapping file not found: {label_mapping_csv}")

    mapping = pd.read_csv(label_mapping_csv)
    if id_col not in mapping.columns:
        raise KeyError(f"Label mapping must contain '{id_col}' column.")
    if "label" not in mapping.columns:
        raise KeyError("Label mapping must contain a 'label' column.")

    merged = out.merge(mapping[[id_col, "label"]], on=id_col, how="left")
    merged = merged.rename(columns={"label": "__label__"})
    return merged


def normalize_binary_labels(series: pd.Series) -> pd.Series:
    # Converts common label forms to {0,1}. Leaves numeric 0/1 as-is.
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(float) > 0).astype("int64")

    s_str = s.astype(str).str.strip().str.lower()
    pos = {"1", "true", "t", "yes", "y", "malicious", "threat", "insider", "anomaly", "positive"}
    neg = {"0", "false", "f", "no", "n", "benign", "normal", "negative"}

    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s_str.isin(pos)] = 1.0
    out[s_str.isin(neg)] = 0.0

    # If still NaN, try best-effort numeric coercion
    fallback = pd.to_numeric(s_str, errors="coerce")
    out = out.fillna((fallback > 0).astype("float"))

    return out.astype("int64")


def prepare_dataframe(
    df: pd.DataFrame,
    timestamp_col: str,
    id_col: str,
    group_col_candidates: tuple[str, ...],
    label_col: str | None = None,
    feature_mode: str = "minimal",
) -> PreparedData:
    df = ensure_datetime(df, timestamp_col)
    df = add_time_features(df, timestamp_col)
    df = add_basic_text_features(df)

    group_col = pick_group_col(df, group_col_candidates)
    if group_col == "__all__":
        df[group_col] = "__all__"

    # Determine feature columns (exclude id/timestamp/label + group identifiers)
    exclude = {id_col, timestamp_col, "__source_file__", group_col}
    if label_col and label_col in df.columns:
        exclude.add(label_col)
    if "__label__" in df.columns:
        exclude.add("__label__")

    feature_cols = [c for c in df.columns if c not in exclude]

    if feature_mode == "minimal":
        # Keep only engineered numeric signals and simple numeric columns.
        keep = {
            "hour",
            "dayofweek",
            "day",
            "month",
            "size",
            "attachments",
            "content_len",
            "url_len",
        }
        feature_cols = [c for c in feature_cols if c in keep]

    return PreparedData(df=df, group_col=group_col, timestamp_col=timestamp_col, id_col=id_col, feature_cols=feature_cols)


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    X = df[feature_cols]

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@dataclass(frozen=True)
class SequenceBatch:
    X: np.ndarray  # (n_seq, seq_len, n_features)
    y: np.ndarray  # (n_seq,)
    lengths: np.ndarray  # (n_seq,) actual lengths before padding


def create_sequences(
    df: pd.DataFrame,
    *,
    group_col: str,
    timestamp_col: str,
    feature_matrix: np.ndarray,
    label: np.ndarray | None,
    seq_len: int,
    stride: int,
    label_strategy: str = "any_positive",
    anchor: str = "start",
) -> SequenceBatch:
    if seq_len <= 1:
        raise ValueError("seq_len must be >= 2")
    if stride <= 0:
        raise ValueError("stride must be >= 1")

    # Keep aligned indices
    df = df.reset_index(drop=True)

    sequences_X: list[np.ndarray] = []
    sequences_y: list[int] = []
    sequences_len: list[int] = []

    for _, idx in df.sort_values(timestamp_col).groupby(group_col).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        # ensure time order within group
        idx = idx[np.argsort(df.loc[idx, timestamp_col].to_numpy())]

        if len(idx) < 2:
            continue

        if anchor not in {"start", "end"}:
            raise ValueError("anchor must be 'start' or 'end'")

        if anchor == "start":
            start = 0
            while start < len(idx):
                end = min(start + seq_len, len(idx))
                window = idx[start:end]
                Xw = feature_matrix[window]
                L = len(window)

                # pad to seq_len at the end
                if L < seq_len:
                    pad = np.zeros((seq_len - L, Xw.shape[1]), dtype=Xw.dtype)
                    Xw = np.vstack([Xw, pad])

                sequences_X.append(Xw)
                sequences_len.append(L)

                if label is not None:
                    yw = label[window]
                    if label_strategy == "last":
                        sequences_y.append(int(yw[-1]))
                    else:  # any_positive
                        sequences_y.append(int(np.max(yw)))

                start += stride
        else:
            # end-anchored windows: each sequence ends at a particular event.
            # This matches per-event labeling (e.g., "this email is misuse").
            end_pos = 0
            while end_pos < len(idx):
                end_idx = end_pos + 1
                start_idx = max(0, end_idx - seq_len)
                window = idx[start_idx:end_idx]
                Xw = feature_matrix[window]
                L = len(window)

                # pad to seq_len at the beginning (left-pad)
                if L < seq_len:
                    pad = np.zeros((seq_len - L, Xw.shape[1]), dtype=Xw.dtype)
                    Xw = np.vstack([pad, Xw])

                sequences_X.append(Xw)
                sequences_len.append(L)

                if label is not None:
                    yw = label[window]
                    if label_strategy == "last":
                        sequences_y.append(int(yw[-1]))
                    else:  # any_positive
                        sequences_y.append(int(np.max(yw)))

                end_pos += stride

    X_seq = np.stack(sequences_X, axis=0) if sequences_X else np.zeros((0, seq_len, feature_matrix.shape[1]))
    lengths = np.asarray(sequences_len, dtype=np.int64)

    if label is None:
        y_seq = np.zeros((X_seq.shape[0],), dtype=np.int64)
    else:
        y_seq = np.asarray(sequences_y, dtype=np.int64)

    return SequenceBatch(X=X_seq.astype(np.float32), y=y_seq, lengths=lengths)
