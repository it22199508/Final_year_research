from __future__ import annotations
import pandas as pd
import numpy as np

PROFILES = {
    # higher percentiles => fewer alerts (more conservative)
    "Conservative": {"t": 0.97, "r": 0.97, "p": 0.93},
    "Default":      {"t": 0.95, "r": 0.95, "p": 0.90},
    # lower percentiles => more alerts (more sensitive)
    "Aggressive":   {"t": 0.90, "r": 0.90, "p": 0.85},
}

def _percentile(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return 0.0
    return float(np.quantile(s, q))

def categorize_threats(df: pd.DataFrame, profile: str = "Default") -> pd.DataFrame:
    out = df.copy()
    cfg = PROFILES.get(profile, PROFILES["Default"])

    # Coerce numeric
    for c in ["temporal_score", "relational_score", "xgb_proba"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Derive time fields
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], errors="coerce")
        out["timestamp"] = ts
        out["date"] = ts.dt.date
        out["hour"] = ts.dt.hour.fillna(0).astype(int)

    t_hi = _percentile(out.get("temporal_score", pd.Series(dtype=float)), cfg["t"])
    r_hi = _percentile(out.get("relational_score", pd.Series(dtype=float)), cfg["r"])
    p_hi = _percentile(out.get("xgb_proba", pd.Series(dtype=float)), cfg["p"]) if "xgb_proba" in out.columns else None

    temporal = out.get("temporal_score", pd.Series(0.0, index=out.index)).fillna(0.0)
    relational = out.get("relational_score", pd.Series(0.0, index=out.index)).fillna(0.0)
    hour = out.get("hour", pd.Series(0, index=out.index)).fillna(0).astype(int)
    odd_hour = (hour <= 5) | (hour >= 22)

    # Threat mapping rules
    is_ato = (temporal >= t_hi) & (relational >= r_hi)        # takeover: drift + unusual access
    if p_hi is not None:
        proba = out["xgb_proba"].fillna(0.0)
        is_ato = is_ato | (proba >= max(0.8, p_hi))           # supervised model boost

    is_priv = (relational >= r_hi) & (~is_ato)                # privilege abuse: unusual relationship edges
    is_susp = (temporal >= t_hi) & odd_hour & (~is_ato) & (~is_priv)  # suspicious login: temporal + odd hour

    out["threat_category"] = "Normal"
    out.loc[is_susp, "threat_category"] = "Suspicious Login"
    out.loc[is_priv, "threat_category"] = "Privilege Abuse"
    out.loc[is_ato, "threat_category"] = "Account Takeover"
    return out

def summary_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[col].astype(str).fillna("unknown").value_counts().rename_axis(col).reset_index(name="count")
