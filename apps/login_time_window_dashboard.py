from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src directory to path for insider_gru module
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root / "src"))

from insider_gru.data import ensure_datetime, load_event_csvs


st.set_page_config(page_title="Device Login Time Window", layout="wide")


def load_device_events(dataset_root: Path, glob_pattern: str) -> pd.DataFrame:
    paths = sorted(dataset_root.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No CSVs matched: {dataset_root}/{glob_pattern}")
    df = load_event_csvs(paths)
    required = {"id", "date", "user", "pc", "activity"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    df = ensure_datetime(df, "date")
    return df


def main() -> None:
    st.title("Device Login Detection (Time Window Rule)")

    st.sidebar.header("Data")
    dataset_root = Path(st.sidebar.text_input("Dataset root", value="Dataset"))

    split = st.sidebar.selectbox(
        "Split",
        options=["train (Dataset/train/R*)", "test (Dataset/test/R*)"],
        index=0,
    )
    if split.startswith("train"):
        glob_pattern = st.sidebar.text_input("Device glob", value="train/R*/device-test-data*.csv")
    else:
        glob_pattern = st.sidebar.text_input("Device glob", value="test/R*/device-train-data.csv")

    st.sidebar.header("Rule")
    activity = st.sidebar.text_input("Login activity", value="Connect")
    start_hour = st.sidebar.number_input("Start hour", min_value=0, max_value=23, value=12, step=1)
    end_hour = st.sidebar.number_input("End hour", min_value=1, max_value=24, value=20, step=1)

    top_k = st.sidebar.slider("Top K (users/PCs)", min_value=12, max_value=20, value=20, step=1)

    try:
        df = load_device_events(dataset_root, glob_pattern)
    except Exception as e:
        st.error(str(e))
        st.stop()

    act = df["activity"].astype(str).str.strip().str.lower()
    is_login = act == str(activity).strip().lower()

    hour = df["date"].dt.hour
    start = int(start_hour)
    end = int(end_hour)
    if start < end:
        in_window = (hour >= start) & (hour < end)
    else:
        in_window = (hour >= start) | (hour < end)

    suspicious = df[is_login & (~in_window)].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total device events", int(len(df)))
    c2.metric("Login events", int(is_login.sum()))
    c3.metric("Suspicious logins (rule)", int(len(suspicious)))

    st.subheader("Top suspicious users")
    users = (
        suspicious.groupby("user")
        .size()
        .sort_values(ascending=False)
        .head(int(top_k))
        .reset_index(name="bad_login_count")
    )
    st.dataframe(users, use_container_width=True)

    st.subheader("Top suspicious PCs")
    pcs = (
        suspicious.groupby("pc")
        .size()
        .sort_values(ascending=False)
        .head(int(top_k))
        .reset_index(name="bad_login_count")
    )
    st.dataframe(pcs, use_container_width=True)

    st.subheader("Bad login events (sample)")
    show_n = st.slider("Rows", min_value=50, max_value=1000, value=200, step=50)
    cols = ["date", "user", "pc", "activity", "id"]
    st.dataframe(suspicious.sort_values("date")[cols].head(int(show_n)), use_container_width=True)


if __name__ == "__main__":
    main()
