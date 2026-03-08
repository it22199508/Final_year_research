"""Explainable feature engineering for insider threat event logs.

This module focuses on simple, deterministic, and report-friendly behavioral
features for research dashboards and sequence modeling.

Supported event types:
- Email events (sender/recipient, attachments, content size, recent sending rate)
- Login/device events (user, device, login timing, recent login rate)

Dependencies: pandas, numpy, typing (standard library).

Design goals
------------
- Keep the original input DataFrame unchanged (functions return copies).
- Prefer transparent rules over complex embeddings/latent features.
- Use safe defaults when optional columns are missing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_BUSINESS_HOURS_START = 8
DEFAULT_BUSINESS_HOURS_END = 18

# Recent-window used for "recent" frequency features.
# Chosen to be easy to explain in a report: "past 24 hours".
DEFAULT_RECENT_WINDOW_HOURS = 24


def _window_str_from_hours(recent_window_hours: int) -> str:
    """Convert an integer hour window into a pandas rolling window string."""

    if not isinstance(recent_window_hours, int):
        raise TypeError("recent_window_hours must be an int")
    if recent_window_hours <= 0:
        raise ValueError("recent_window_hours must be > 0")
    return f"{recent_window_hours}h"


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """Parse timestamps safely; malformed values become NaT."""

    return pd.to_datetime(series, errors="coerce")


def _get_business_hours_from_df(
    df: pd.DataFrame,
    business_hours_start: Optional[int] = None,
    business_hours_end: Optional[int] = None,
) -> tuple[int, int]:
    """Resolve business hours from explicit args, df.attrs, or module defaults."""

    start = business_hours_start
    end = business_hours_end

    if start is None:
        start = int(df.attrs.get("business_hours_start", DEFAULT_BUSINESS_HOURS_START))
    if end is None:
        end = int(df.attrs.get("business_hours_end", DEFAULT_BUSINESS_HOURS_END))

    # Clamp into [0, 23] and keep deterministic behavior on weird inputs.
    start = int(max(0, min(23, start)))
    end = int(max(0, min(24, end)))
    return start, end


def _after_hours_flag(hour: pd.Series, business_hours_start: int, business_hours_end: int) -> pd.Series:
    """Return 1 for hours outside the business window, else 0.

    Treats business hours as a half-open interval [start, end).
    """

    # hour can be float due to NaNs; comparisons still work.
    after = (hour < business_hours_start) | (hour >= business_hours_end)
    return after.fillna(False).astype(int)


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Add basic time-derived features.

    Adds:
        - hour_of_day: 0..23 (NaN if timestamp invalid)
        - day_of_week: 0..6 where Monday=0 (NaN if timestamp invalid)
        - is_weekend: 1 if Saturday/Sunday else 0
        - is_after_hours: 1 if outside business hours else 0

    Business hours configuration:
        This function reads optional values from `df.attrs`:
            - df.attrs['business_hours_start'] (default 8)
            - df.attrs['business_hours_end']   (default 18)

        This keeps the signature simple while still allowing configuration.

    Args:
        df: Input DataFrame.
        timestamp_col: Name of the timestamp column.

    Returns:
        A copy of the DataFrame with additional time features.

    Raises:
        ValueError: If timestamp_col is missing.
    """

    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    # Deep copy to avoid mutating caller-owned columns/blocks unexpectedly.
    out = df.copy(deep=True)

    start, end = _get_business_hours_from_df(out)

    ts = _safe_to_datetime(out[timestamp_col])
    hour = ts.dt.hour.astype("float")
    dow = ts.dt.dayofweek.astype("float")

    out["hour_of_day"] = hour
    out["day_of_week"] = dow
    out["is_weekend"] = (dow >= 5).fillna(False).astype(int)
    out["is_after_hours"] = _after_hours_flag(hour, start, end)

    return out


def extract_domain(email: str) -> str:
    """Extract a lowercase domain from an email-like string.

    Returns an empty string for malformed inputs.

    Examples:
        - "User@Example.com" -> "example.com"
        - "not-an-email" -> ""

    Args:
        email: Email address or email-like value.

    Returns:
        Lowercase domain, or "".
    """

    if email is None:
        return ""

    try:
        s = str(email).strip().lower()
    except Exception:
        return ""

    if "@" not in s:
        return ""

    domain = s.split("@")[-1].strip()
    # Basic sanity checks: no spaces and has a dot.
    if not domain or " " in domain or "." not in domain:
        return ""

    return domain


def flag_external_recipient(sender: str, recipient: str) -> int:
    """Flag if recipient domain differs from sender domain.

    Returns 0 if either side is missing or malformed.

    Args:
        sender: Sender email.
        recipient: Recipient email.

    Returns:
        1 if external recipient relative to sender else 0.
    """

    sender_domain = extract_domain(sender)
    recipient_domain = extract_domain(recipient)

    if not sender_domain or not recipient_domain:
        return 0

    return int(sender_domain != recipient_domain)


def _coerce_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    """Convert a series to numeric safely, filling invalid with default."""

    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(default)


def _compute_recent_counts(
    df: pd.DataFrame,
    group_col: str,
    timestamp_col: str,
    window: str = f"{DEFAULT_RECENT_WINDOW_HOURS}h",
) -> pd.Series:
    """Compute per-group event counts in a trailing time window (excluding current).

    Returns a Series aligned to df.index.
    """

    ts = _safe_to_datetime(df[timestamp_col])

    temp = pd.DataFrame(
        {
            "__orig_index": df.index,
            "_ts": ts,
            "_group": df[group_col].astype(str),
            "_one": 1,
        },
        index=df.index,
    )

    if temp["_ts"].notna().sum() == 0:
        return pd.Series(0, index=df.index, dtype=int)

    temp = temp.sort_values(["_group", "_ts"], kind="mergesort")
    temp = temp.set_index("_ts")

    rolled = (
        temp.groupby("_group")["_one"]
        .rolling(window=window, closed="left")
        .sum()
        .reset_index(level=0, drop=True)
    )

    temp["_recent_count"] = rolled.to_numpy()
    out = (
        temp.reset_index(drop=False)
        .set_index("__orig_index")["_recent_count"]
        .reindex(df.index)
        .fillna(0)
        .astype(int)
    )

    return out


def _compute_recent_unique(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    timestamp_col: str,
    window: str = f"{DEFAULT_RECENT_WINDOW_HOURS}h",
) -> pd.Series:
    """Compute per-group unique counts in a trailing time window (excluding current).

    Uses a simple pandas rolling-apply. This is explainable but can be slower on
    very large datasets.
    """

    ts = _safe_to_datetime(df[timestamp_col])
    values = df[value_col].astype(str)
    value_codes = pd.Categorical(values).codes  # -1 indicates NaN/unknown

    temp = pd.DataFrame(
        {
            "__orig_index": df.index,
            "_ts": ts,
            "_group": df[group_col].astype(str),
            "_value_code": value_codes,
        },
        index=df.index,
    )

    if temp["_ts"].notna().sum() == 0:
        return pd.Series(0, index=df.index, dtype=int)

    temp = temp.sort_values(["_group", "_ts"], kind="mergesort")
    temp = temp.set_index("_ts")

    def _nunique_codes(arr: np.ndarray) -> float:
        # arr may be float if pandas upcasts; keep -1 as missing marker.
        arr_int = arr.astype(int, copy=False)
        arr_int = arr_int[arr_int != -1]
        if arr_int.size == 0:
            return 0.0
        return float(np.unique(arr_int).size)

    rolled = (
        temp.groupby("_group")["_value_code"]
        .rolling(window=window, closed="left")
        .apply(_nunique_codes, raw=True)
        .reset_index(level=0, drop=True)
    )

    temp["_recent_unique"] = rolled.to_numpy()
    out = (
        temp.reset_index(drop=False)
        .set_index("__orig_index")["_recent_unique"]
        .reindex(df.index)
        .fillna(0)
        .astype(int)
    )

    return out


def engineer_email_features(
    df: pd.DataFrame,
    timestamp_col: str = "shared_time",
    business_hours_start: int = 8,
    business_hours_end: int = 18,
    recent_window_hours: int = DEFAULT_RECENT_WINDOW_HOURS,
) -> pd.DataFrame:
    """Engineer explainable behavioral features for email events.

    Output features include:
        - hour_of_day, day_of_week, is_weekend, is_after_hours
        - attachment_count, content_size, has_attachment
        - external_recipient_flag
        - sent_count_recent_window (per sender, trailing recent_window_hours excluding current)
        - unique_recipients_recent_window (per sender, trailing recent_window_hours excluding current)

    Expected (optional) input columns:
        - sender (or from)
        - recipient (or to)
        - attachment_count (or attachments)
        - content_size (or size)

    Args:
        df: Input DataFrame of email events.
        timestamp_col: Timestamp column name.
        business_hours_start: Start hour for business-hours window.
        business_hours_end: End hour for business-hours window.
        recent_window_hours: Size of recent window in hours for rolling features.

    Returns:
        A copy of the DataFrame with engineered features.

    Raises:
        ValueError: If timestamp_col is missing.
    """

    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    # Deep copy: this function may overwrite existing columns (e.g., attachment_count).
    out = df.copy(deep=True)
    out.attrs = dict(df.attrs)
    out.attrs["business_hours_start"] = business_hours_start
    out.attrs["business_hours_end"] = business_hours_end

    out = add_time_features(out, timestamp_col)

    window = _window_str_from_hours(recent_window_hours)

    # Column normalization: support common email column aliases.
    sender_col = "sender" if "sender" in out.columns else ("from" if "from" in out.columns else None)
    recipient_col = "recipient" if "recipient" in out.columns else ("to" if "to" in out.columns else None)

    if sender_col is None:
        out["sender"] = ""
        sender_col = "sender"
    if recipient_col is None:
        out["recipient"] = ""
        recipient_col = "recipient"

    # attachment_count
    if "attachment_count" in out.columns:
        attachment_count = _coerce_numeric(out["attachment_count"], default=0)
    elif "attachments" in out.columns:
        # If attachments is a list-like string, count separators; otherwise fall back to 0/1.
        att = out["attachments"].astype(str).fillna("")
        # Very simple heuristic: count commas and add 1 if non-empty.
        attachment_count = att.apply(lambda s: 0 if not s or s.lower() in {"nan", "none"} else (s.count(",") + 1))
        attachment_count = _coerce_numeric(attachment_count, default=0)
    else:
        attachment_count = pd.Series(0, index=out.index, dtype=float)

    out["attachment_count"] = attachment_count.astype(int)
    out["has_attachment"] = (out["attachment_count"] > 0).astype(int)

    # content_size
    if "content_size" in out.columns:
        content_size = _coerce_numeric(out["content_size"], default=0)
    elif "size" in out.columns:
        content_size = _coerce_numeric(out["size"], default=0)
    else:
        content_size = pd.Series(0, index=out.index, dtype=float)

    out["content_size"] = content_size.astype(float)

    # External recipient flag
    out["external_recipient_flag"] = [
        flag_external_recipient(s, r)
        for s, r in zip(out[sender_col].astype(str), out[recipient_col].astype(str), strict=False)
    ]

    # Recent window features (per sender)
    out["sent_count_recent_window"] = _compute_recent_counts(
        out,
        group_col=sender_col,
        timestamp_col=timestamp_col,
        window=window,
    )

    out["unique_recipients_recent_window"] = _compute_recent_unique(
        out,
        group_col=sender_col,
        value_col=recipient_col,
        timestamp_col=timestamp_col,
        window=window,
    )

    return out


def engineer_login_features(
    df: pd.DataFrame,
    timestamp_col: str = "login_time",
    business_hours_start: int = 8,
    business_hours_end: int = 18,
    recent_window_hours: int = DEFAULT_RECENT_WINDOW_HOURS,
) -> pd.DataFrame:
    """Engineer explainable behavioral features for login/device events.

    Output features include:
        - login_hour
        - day_of_week, is_weekend
        - after_hours_login
        - login_frequency_recent_window (per user, trailing recent_window_hours excluding current)
        - unusual_device_flag (if device column exists): 1 if device not seen before for user

    Expected (optional) input columns:
        - user (or username)
        - device

    Args:
        df: Input DataFrame of login/device events.
        timestamp_col: Timestamp column name.
        business_hours_start: Start hour for business-hours window.
        business_hours_end: End hour for business-hours window.
        recent_window_hours: Size of recent window in hours for rolling features.

    Returns:
        A copy of the DataFrame with engineered features.

    Raises:
        ValueError: If timestamp_col is missing.
    """

    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    # Deep copy: this function adds and may overwrite feature columns.
    out = df.copy(deep=True)
    out.attrs = dict(df.attrs)
    out.attrs["business_hours_start"] = business_hours_start
    out.attrs["business_hours_end"] = business_hours_end

    # Standardize time features with the shared helper.
    # This produces: hour_of_day, day_of_week, is_weekend, is_after_hours
    out.attrs = dict(out.attrs)
    out.attrs["business_hours_start"] = business_hours_start
    out.attrs["business_hours_end"] = business_hours_end
    out = add_time_features(out, timestamp_col)

    # Backwards-compatible aliases commonly used in login-specific reports.
    out["login_hour"] = out["hour_of_day"]
    out["after_hours_login"] = out["is_after_hours"]

    window = _window_str_from_hours(recent_window_hours)

    user_col = "user" if "user" in out.columns else ("username" if "username" in out.columns else None)
    if user_col is None:
        out["user"] = ""
        user_col = "user"

    out["login_frequency_recent_window"] = _compute_recent_counts(
        out,
        group_col=user_col,
        timestamp_col=timestamp_col,
        window=window,
    )

    # Unusual device flag: first-seen device for a user.
    if "device" in out.columns:
        temp = out[[user_col, "device", timestamp_col]].copy(deep=False)
        temp["_ts"] = _safe_to_datetime(temp[timestamp_col])
        temp["_user"] = temp[user_col].astype(str)
        temp["_device"] = temp["device"].astype(str)

        temp = temp.sort_values(["_user", "_ts"], kind="mergesort")
        # Flag as 1 if this is a new (user, device) combination AND the user has prior history.
        # This avoids marking the very first login for a user as "unusual".
        is_first_event_for_user = temp.groupby("_user").cumcount() == 0
        is_new_device_for_user = ~temp.duplicated(subset=["_user", "_device"], keep="first")
        unusual = is_new_device_for_user & (~is_first_event_for_user)
        out["unusual_device_flag"] = unusual.reindex(out.index).fillna(False).astype(int)
    else:
        out["unusual_device_flag"] = 0

    return out


def build_behavioral_features(
    df: pd.DataFrame,
    event_type: str,
    timestamp_col: str,
    recent_window_hours: int = DEFAULT_RECENT_WINDOW_HOURS,
) -> pd.DataFrame:
    """Dispatch to the correct feature engineering pipeline.

    Args:
        df: Input DataFrame.
        event_type: "email" or "login".
        timestamp_col: Timestamp column for the selected event type.
        recent_window_hours: Size of recent window in hours for rolling features.

    Returns:
        DataFrame with engineered features.

    Raises:
        ValueError: For unsupported event_type.
    """

    et = (event_type or "").strip().lower()

    if et in {"email", "emails"}:
        return engineer_email_features(
            df,
            timestamp_col=timestamp_col,
            recent_window_hours=recent_window_hours,
        )
    if et in {"login", "logins", "device", "auth", "authentication"}:
        return engineer_login_features(
            df,
            timestamp_col=timestamp_col,
            recent_window_hours=recent_window_hours,
        )

    raise ValueError(
        f"Unsupported event_type: {event_type!r}. Supported types: 'email', 'login'."
    )
