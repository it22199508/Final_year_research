from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


DEVICE_LOG_COLUMNS = ["event_id", "timestamp", "user", "pc", "activity"]


def load_device_log_from_path(path: str) -> pd.DataFrame:
    """Load device log CSV from a local filesystem path."""

    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"CSV not found: {p}")
    return _load_device_log(p)


def discover_test_folders(base_path: str | Path) -> list[str]:
    """Discover available test groups (e.g., R1..R4) under Dataset/test.

    Returns folder names (not full paths) sorted by numeric suffix when possible.
    """

    base = Path(base_path)
    if not base.exists() or not base.is_dir():
        return []

    groups: list[str] = []
    for child in base.iterdir():
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


def find_device_csv_files(folder_path: str | Path) -> list[Path]:
    """Find likely device log CSV files inside a group folder.

    The project dataset folders can include multiple modalities (device/email/http).
    For after-hours login detection we prefer files whose names include "device".
    """

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []

    # Prefer device-related CSVs only. Some datasets use legacy "logon" in the filename.
    csv_paths = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    ]
    matches = [
        p
        for p in csv_paths
        if ("device" in p.name.lower()) or ("logon" in p.name.lower())
    ]
    return sorted(matches)


def load_combined_device_logs_from_test(
    base_path: str | Path,
    selected_group: str | None = None,
) -> pd.DataFrame:
    """Load and combine device log CSV files from Dataset/test.

    Parameters
    - base_path: path to Dataset/test
    - selected_group: None for all groups, or a specific group like "R1".

    Returns a combined raw dataframe with added columns:
      - group: R1/R2/...
      - source_file: filename
      - source_path: full path
    """

    base = Path(base_path)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Test base path not found: {base}")

    available_groups = discover_test_folders(base)
    if not available_groups:
        raise FileNotFoundError(f"No R-folders found under: {base}")

    groups_to_load: list[str]
    if selected_group is None:
        groups_to_load = available_groups
    else:
        selected_group = selected_group.strip()
        if selected_group not in available_groups:
            raise ValueError(f"Selected group {selected_group!r} not found under {base}")
        groups_to_load = [selected_group]

    frames: list[pd.DataFrame] = []
    for g in groups_to_load:
        folder = base / g
        files = find_device_csv_files(folder)
        for p in files:
            df = load_device_log_from_path(str(p))
            df["group"] = g
            df["source_file"] = p.name
            df["source_path"] = str(p)
            frames.append(df)

    if not frames:
        groups_msg = ", ".join(groups_to_load)
        raise FileNotFoundError(
            f"No device CSV files found under {base} for groups: {groups_msg}. "
            "Expected filenames containing 'device' or 'logon' (case-insensitive)."
        )

    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_connect_events_from_test(
    test_base_path: str,
    selected_group: str | None,
) -> pd.DataFrame:
    """Load, parse, and filter to Connect events for the chosen test group(s).

    This is cached to keep the dashboard responsive when users adjust filters.
    """

    raw_df = load_combined_device_logs_from_test(test_base_path, selected_group=selected_group)
    prepared = prepare_device_log(raw_df)
    return filter_connect_events(prepared)


def _load_device_log(source: Any) -> pd.DataFrame:
    """Internal CSV loader (headerless CSV)."""

    df = pd.read_csv(
        source,
        header=None,
        dtype=str,
        on_bad_lines="skip",
        encoding_errors="replace",
    )

    if df.shape[1] < len(DEVICE_LOG_COLUMNS):
        raise ValueError(
            f"CSV has too few columns ({df.shape[1]}). Expected at least {len(DEVICE_LOG_COLUMNS)} columns: "
            f"{DEVICE_LOG_COLUMNS}"
        )

    # If extra columns exist, keep only the first 5.
    df = df.iloc[:, : len(DEVICE_LOG_COLUMNS)].copy()
    df.columns = DEVICE_LOG_COLUMNS
    return df


def prepare_device_log(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich device log data.

    - Strips whitespace from string columns
    - Parses timestamp into `timestamp_dt`
    - Drops invalid timestamps
    - Creates:
        login_date, login_time, hour, day_of_week, is_weekend
    """

    out = df.copy()

    for c in DEVICE_LOG_COLUMNS:
        if c not in out.columns:
            raise ValueError(f"Missing expected column: {c!r}")
        out[c] = out[c].astype(str).str.strip()

    # Timestamp parsing.
    # Example: 01/04/2010 07:12:31
    # Most CERT-style datasets use MM/DD/YYYY.
    ts_raw = out["timestamp"]
    ts1 = pd.to_datetime(ts_raw, format="%m/%d/%Y %H:%M:%S", errors="coerce")
    nat_rate = float(ts1.isna().mean()) if len(ts1) else 1.0

    if nat_rate > 0.5:
        ts2 = pd.to_datetime(ts_raw, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        if float(ts2.isna().mean()) < nat_rate:
            out["timestamp_dt"] = ts2
        else:
            out["timestamp_dt"] = ts1
    else:
        out["timestamp_dt"] = ts1

    out = out[out["timestamp_dt"].notna()].copy()

    out["login_date"] = out["timestamp_dt"].dt.date
    out["login_time"] = out["timestamp_dt"].dt.time
    out["hour"] = out["timestamp_dt"].dt.hour
    out["day_of_week"] = out["timestamp_dt"].dt.day_name()
    out["is_weekend"] = out["timestamp_dt"].dt.dayofweek >= 5

    return out


def filter_connect_events(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where activity == 'Connect' (case-insensitive)."""

    if "activity" not in df.columns:
        return df.iloc[0:0].copy()
    mask = df["activity"].astype(str).str.strip().str.lower().eq("connect")
    return df[mask].copy()


def filter_by_date_range(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """Filter by date range (inclusive) using the `login_date` field."""

    if "login_date" not in df.columns:
        raise ValueError("filter_by_date_range requires 'login_date'; call prepare_device_log() first")
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    return df[(df["login_date"] >= start_date) & (df["login_date"] <= end_date)].copy()


def flag_after_hours_logins(
    df: pd.DataFrame,
    office_start: time,
    office_end: time,
) -> pd.DataFrame:
    """Flag whether each Connect event is outside office hours.

    After-hours definition:
    - time before office_start OR time after office_end
    """

    if "login_time" not in df.columns:
        raise ValueError("flag_after_hours_logins requires 'login_time'; call prepare_device_log() first")

    out = df.copy()
    t = out["login_time"]

    # Within hours is defined as office_start <= time <= office_end.
    within = (t >= office_start) & (t <= office_end)
    after_hours = ~within

    out["status"] = after_hours.map(lambda x: "After Hours" if bool(x) else "Within Hours")
    return out


def summarize_after_hours_by_user(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize after-hours logins grouped by user.

    Groups only flagged rows (status == "After Hours").
    Returns:
      user, after_hours_login_count, first_after_hours_login, last_after_hours_login, unique_pcs_used
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "user",
                "after_hours_login_count",
                "first_after_hours_login",
                "last_after_hours_login",
                "unique_pcs_used",
            ]
        )

    required = {"user", "pc", "timestamp_dt", "status"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise ValueError(f"summarize_after_hours_by_user missing required columns: {missing}")

    flagged = df[df["status"] == "After Hours"].copy()
    if flagged.empty:
        return pd.DataFrame(
            columns=[
                "user",
                "after_hours_login_count",
                "first_after_hours_login",
                "last_after_hours_login",
                "unique_pcs_used",
            ]
        )

    g = flagged.groupby("user", dropna=False)
    summary = pd.DataFrame(
        {
            "after_hours_login_count": g.size(),
            "first_after_hours_login": g["timestamp_dt"].min(),
            "last_after_hours_login": g["timestamp_dt"].max(),
            "unique_pcs_used": g["pc"].nunique(dropna=True),
        }
    ).reset_index()

    return summary.sort_values("after_hours_login_count", ascending=False, na_position="last")


def main() -> None:
    st.set_page_config(page_title="After-Hours Login Detection Dashboard", layout="wide")

    st.title("After-Hours Login Detection Dashboard")
    st.caption("Uses fixed dataset structure (test split only): Dataset/test/R1..R4")

    test_base_path = str(Path("Dataset") / "test")

    # --- Sidebar controls ---
    with st.sidebar:
        st.header("Detection Settings")
        st.caption("No CSV upload; the app scans Dataset/test automatically.")

        groups = discover_test_folders(test_base_path)
        if not groups:
            st.error(f"No test groups found under: {test_base_path}")
            group_choice = "All"
            selected_group = None
            can_run = False
            preview_df = pd.DataFrame()
            min_date = date.today()
            max_date = date.today()
        else:
            group_options = ["All"] + groups
            group_choice = st.selectbox(
                "Test group",
                options=group_options,
                index=0,
                help="Choose All test groups or a specific group (R1..R4).",
            )
            selected_group = None if group_choice == "All" else group_choice
            try:
                preview_df = load_connect_events_from_test(test_base_path, selected_group)
                if not preview_df.empty:
                    min_date = preview_df["login_date"].min()
                    max_date = preview_df["login_date"].max()
                else:
                    min_date = date.today()
                    max_date = date.today()
                can_run = True
            except Exception as e:
                st.error(f"Failed to load test data: {type(e).__name__}: {e}")
                preview_df = pd.DataFrame()
                min_date = date.today()
                max_date = date.today()
                can_run = False

        office_start = st.time_input("Office start time", value=time(8, 0))
        office_end = st.time_input("Office end time", value=time(18, 0))

        start_date = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )
        end_date = st.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

        if start_date > end_date:
            st.error("Start date must be <= End date")
            can_run = False

        run_clicked = st.button("Run Detection", type="primary", disabled=not can_run)

    # Persistent results
    if "results" not in st.session_state:
        st.session_state["results"] = {}

    if run_clicked:
        try:
            connect_df = load_connect_events_from_test(test_base_path, selected_group)
            connect_df = filter_by_date_range(connect_df, start_date=start_date, end_date=end_date)
            source_label = f"{Path(test_base_path)} ({'ALL' if selected_group is None else selected_group})"

            processed = flag_after_hours_logins(connect_df, office_start=office_start, office_end=office_end)

            flagged = processed[processed["status"] == "After Hours"].copy()
            flagged = flagged.sort_values("timestamp_dt", ascending=False, na_position="last")

            summary = summarize_after_hours_by_user(processed)
            counts_by_user = (
                summary[["user", "after_hours_login_count"]].copy()
                if not summary.empty
                else pd.DataFrame(columns=["user", "after_hours_login_count"])
            )

            st.session_state["results"] = {
                "source": source_label,
                "processed": processed,
                "flagged": flagged,
                "summary": summary,
                "counts": counts_by_user,
                "settings": {
                    "office_start": office_start.isoformat(timespec="minutes"),
                    "office_end": office_end.isoformat(timespec="minutes"),
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                },
            }
        except Exception as e:
            st.session_state["results"] = {}
            st.error(f"Detection failed: {type(e).__name__}: {e}")

    results = st.session_state.get("results") or {}
    if not results:
        st.info("Configure settings in the sidebar, then click **Run Detection**.")
        return

    processed_df: pd.DataFrame = results.get("processed", pd.DataFrame())
    flagged_df: pd.DataFrame = results.get("flagged", pd.DataFrame())
    summary_df: pd.DataFrame = results.get("summary", pd.DataFrame())

    st.subheader("Flagged event table")
    if flagged_df.empty:
        st.info("No after-hours logins were flagged for the selected settings/date range.")
    else:
        view = flagged_df.copy().rename(columns={"timestamp_dt": "timestamp"})
        cols = ["event_id", "timestamp", "user", "pc", "activity"]
        view = view[[c for c in cols if c in view.columns]]
        st.dataframe(view, use_container_width=True, height=420)

    st.subheader("Grouped summary by user")
    if summary_df.empty:
        st.info("No users were flagged.")
    else:
        st.dataframe(summary_df, use_container_width=True, height=360)

    st.subheader("Counts by user")
    counts_df: pd.DataFrame = results.get("counts", pd.DataFrame())
    if counts_df.empty:
        st.info("No after-hours login counts to display.")
    else:
        st.dataframe(counts_df, use_container_width=True, height=300)


if __name__ == "__main__":
    main()
