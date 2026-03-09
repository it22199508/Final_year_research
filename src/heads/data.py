from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED_COLS = [
    "timestamp","src_ip","dst_ip","src_port","dst_port","protocol",
    "bytes_sent","bytes_received","user_agent","url","is_internal_traffic"
]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=False)
    if df["timestamp"].isna().any():
        # keep but warn via NaT; later sorted
        pass

    # clean types
    for c in ["src_port","dst_port","bytes_sent","bytes_received"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["protocol"] = df["protocol"].astype(str).fillna("UNK")
    df["user_agent"] = df["user_agent"].astype(str).fillna("UNK")
    df["url"] = df["url"].astype(str).replace("nan", "").fillna("")
    df["is_internal_traffic"] = df["is_internal_traffic"].astype(bool)

    # Optional labels
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    if "attack_type" in df.columns:
        df["attack_type"] = df["attack_type"].astype(str).fillna("benign")

    # sort for sequence modelling
    df = df.sort_values(["src_ip","timestamp"], kind="mergesort").reset_index(drop=True)
    return df
