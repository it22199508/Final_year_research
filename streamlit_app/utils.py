from __future__ import annotations
import pandas as pd
import numpy as np
import re
from pathlib import Path

REQUIRED_COLS = [
    "timestamp","src_ip","dst_ip","src_port","dst_port","protocol",
    "bytes_sent","bytes_received","user_agent","url","is_internal_traffic"
]

def extract_url_host(url: str) -> str:
    if not url or str(url).lower() == "nan":
        return ""
    m = re.match(r"^https?://([^/]+)/?", str(url))
    return m.group(1).lower() if m else ""

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["hour"] = out["timestamp"].dt.hour.fillna(0).astype(int)
    out["dow"] = out["timestamp"].dt.dayofweek.fillna(0).astype(int)
    if "url" not in out.columns:
        out["url"] = ""
    if "user_agent" not in out.columns:
        out["user_agent"] = "UNK"
    if "protocol" not in out.columns:
        out["protocol"] = "UNK"
    out["url_host"] = out["url"].apply(extract_url_host)

    for c in ["bytes_sent","bytes_received","src_port","dst_port"]:
        if c not in out.columns:
            out[c] = 0
    out["bytes_sent"] = pd.to_numeric(out["bytes_sent"], errors="coerce").fillna(0).astype(int)
    out["bytes_received"] = pd.to_numeric(out["bytes_received"], errors="coerce").fillna(0).astype(int)
    out["bytes_total"] = (out["bytes_sent"] + out["bytes_received"]).astype(int)
    out["src_port"] = pd.to_numeric(out["src_port"], errors="coerce").fillna(0).astype(int)
    out["dst_port"] = pd.to_numeric(out["dst_port"], errors="coerce").fillna(0).astype(int)

    out["protocol"] = out["protocol"].astype(str).fillna("UNK")
    out["user_agent"] = out["user_agent"].astype(str).fillna("UNK")
    out["url"] = out["url"].astype(str).replace("nan","").fillna("")

    if "is_internal_traffic" not in out.columns:
        out["is_internal_traffic"] = False
    out["is_internal_traffic"] = out["is_internal_traffic"].astype(bool).astype(int)
    out["is_web"] = out["dst_port"].isin([80,443]).astype(int)
    return out

def validate_df_basic(df: pd.DataFrame):
    issues=[]
    missing_cols=[c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    summary={
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for c in ["src_port","dst_port"]:
        if c in df.columns:
            s=pd.to_numeric(df[c], errors="coerce")
            bad=int(((s<0)|(s>65535)).sum())
            if bad:
                issues.append(f"{c}: {bad} values outside 0..65535")

    for c in ["bytes_sent","bytes_received"]:
        if c in df.columns:
            s=pd.to_numeric(df[c], errors="coerce")
            bad=int((s<0).sum())
            if bad:
                issues.append(f"{c}: {bad} negative values")

    if "timestamp" in df.columns:
        ts=pd.to_datetime(df["timestamp"], errors="coerce")
        bad=float(ts.isna().mean())
        summary["bad_timestamp_pct"]=bad*100.0
        if bad>0.01:
            issues.append(f"timestamp: {bad*100:.2f}% unparseable")

    return len(issues)==0, issues, summary

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
