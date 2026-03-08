from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable, Tuple

import pandas as pd


DEVICE_LOG_COLUMNS = ["event_id", "timestamp", "user", "pc", "activity"]


@dataclass(frozen=True)
class OfficeHours:
    """Office hours window.

    Supports standard hours (start < end) and overnight windows (start > end).
    """

    start: time
    end: time

    def contains(self, t: time) -> bool:
        if self.start == self.end:
            return True
        if self.start < self.end:
            return self.start <= t < self.end
        # Overnight window (e.g., 22:00 -> 06:00)
        return t >= self.start or t < self.end


def load_device_log(path: str | Path | Any) -> pd.DataFrame:
    """Load a device log CSV that has *no header row*.

    Expected column order:
        event_id, timestamp, user, pc, activity

    Notes:
    - Reads all columns as strings (keeps IDs stable).
    - Skips malformed rows via pandas' `on_bad_lines="skip"`.

    Parameters
    - path: filesystem path or file-like

    Returns
    - DataFrame with columns: DEVICE_LOG_COLUMNS
    """

    df = pd.read_csv(
        path,
        header=None,
        names=DEVICE_LOG_COLUMNS,
        dtype=str,
        on_bad_lines="skip",
        encoding_errors="replace",
    )

    # Normalize whitespace and "nan"-like strings.
    for c in DEVICE_LOG_COLUMNS:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            s = s.replace({"": None, "nan": None, "None": None})
            df[c] = s

    return df


def prepare_device_log(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a device log DataFrame.

    - Ensures expected columns exist
    - Parses timestamp into a new column `timestamp_dt`

    Timestamp parsing:
    - Tries MM/DD/YYYY HH:MM:SS (common CERT-style)
    - If too many NaT, falls back to DD/MM/YYYY HH:MM:SS

    Returns
    - Copy of df with a `timestamp_dt` datetime64 column
    """

    out = df.copy()
    for c in DEVICE_LOG_COLUMNS:
        if c not in out.columns:
            out[c] = None

    ts_raw = out["timestamp"]

    ts1 = pd.to_datetime(ts_raw, format="%m/%d/%Y %H:%M:%S", errors="coerce")
    nat_rate = float(ts1.isna().mean()) if len(ts1) else 1.0

    if nat_rate > 0.5:
        ts2 = pd.to_datetime(ts_raw, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        if float(ts2.isna().mean()) < nat_rate:
            out["timestamp_dt"] = ts2
            return out

    out["timestamp_dt"] = ts1
    return out


def filter_connect_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where activity == 'Connect' (case-insensitive)."""

    if "activity" not in df.columns:
        return df.iloc[0:0].copy()

    act = df["activity"].astype(str).str.strip().str.lower()
    return df[act.eq("connect")].copy()


def filter_by_date_range(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """Filter rows to an inclusive date range using `timestamp_dt`.

    Rows with invalid timestamps (NaT) are dropped.

    Parameters
    - start_date/end_date: Python `date`

    Returns
    - Filtered DataFrame
    """

    if "timestamp_dt" not in df.columns:
        raise ValueError("filter_by_date_range requires a 'timestamp_dt' column; call prepare_device_log() first")

    out = df[df["timestamp_dt"].notna()].copy()

    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max)

    return out[(out["timestamp_dt"] >= start_dt) & (out["timestamp_dt"] <= end_dt)].copy()


def flag_after_hours_logins(
    df: pd.DataFrame,
    office_start: time,
    office_end: time,
    *,
    timestamp_col: str = "timestamp_dt",
) -> pd.DataFrame:
    """Flag whether each row is outside office hours.

    Adds columns:
    - `within_hours`: bool
    - `status`: "Within Hours" or "After Hours"

    Parameters
    - office_start/office_end: Python `time`
    - timestamp_col: name of datetime column (default: timestamp_dt)

    Returns
    - Copy of df with flags
    """

    if timestamp_col not in df.columns:
        raise ValueError(f"flag_after_hours_logins requires '{timestamp_col}'; call prepare_device_log() first")

    out = df.copy()
    hours = OfficeHours(start=office_start, end=office_end)

    # If timestamps are NaT, mark as not within hours.
    ts = out[timestamp_col]
    times = pd.to_datetime(ts, errors="coerce").dt.time

    within = times.apply(lambda t: hours.contains(t) if isinstance(t, time) else False)
    out["within_hours"] = within.astype(bool)
    out["status"] = out["within_hours"].map(lambda w: "Within Hours" if bool(w) else "After Hours")

    return out


def summarize_after_hours_by_user(
    df_flagged: pd.DataFrame,
    *,
    user_col: str = "user",
    timestamp_col: str = "timestamp_dt",
    status_col: str = "status",
) -> pd.DataFrame:
    """Group after-hours logins by user.

    Returns columns:
    - user
    - after_hours_count
    - first_after_hours_login
    - last_after_hours_login
    """

    if df_flagged.empty:
        return pd.DataFrame(columns=[user_col, "after_hours_count", "first_after_hours_login", "last_after_hours_login"])

    for col in (user_col, timestamp_col, status_col):
        if col not in df_flagged.columns:
            raise ValueError(f"summarize_after_hours_by_user requires column '{col}'")

    after = df_flagged[df_flagged[status_col] == "After Hours"].copy()
    if after.empty:
        return pd.DataFrame(columns=[user_col, "after_hours_count", "first_after_hours_login", "last_after_hours_login"])

    g = after.groupby(user_col, dropna=False)[timestamp_col]
    summary = pd.DataFrame(
        {
            "after_hours_count": g.size(),
            "first_after_hours_login": g.min(),
            "last_after_hours_login": g.max(),
        }
    ).reset_index()

    return summary.sort_values("after_hours_count", ascending=False, na_position="last")


def detect_after_hours_logins(
    path: str | Path | Any,
    *,
    start_date: date,
    end_date: date,
    office_start: time,
    office_end: time,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: load -> prepare -> connect-only -> date-range -> flag -> summarize."""

    df = load_device_log(path)
    df = prepare_device_log(df)
    df = filter_connect_events(df)
    df = filter_by_date_range(df, start_date, end_date)
    flagged = flag_after_hours_logins(df, office_start=office_start, office_end=office_end)
    summary = summarize_after_hours_by_user(flagged)
    return flagged, summary
