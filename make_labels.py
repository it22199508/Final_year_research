import argparse
import glob
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

EPS = 1e-9

def _find_file(root: str, filename: str):
    hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
    return hits[0] if hits else None

def _pick_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # fallback: fuzzy contains
    for cand in candidates:
        for lc, orig in cols.items():
            if cand.lower() in lc:
                return orig
    return None

def _to_datetime_series(s):
    return pd.to_datetime(s, errors="coerce", utc=False)

def _day_col(dt_series):
    # normalize to date (no timezone assumption)
    return dt_series.dt.date.astype(str)

def robust_z(x: pd.Series):
    med = x.median()
    mad = (x - med).abs().median()
    scale = 1.4826 * mad + EPS
    return (x - med) / scale

def load_log(path, kind):
    # 1) try normal CSV with header
    df = pd.read_csv(path)

    def has_user_date(d):
        cols = [c.lower() for c in d.columns]
        has_user = any("user" in c for c in cols)
        has_date = any(c in ("date", "time", "timestamp", "datetime") or "date" in c or "time" in c for c in cols)
        return has_user and has_date

    # 2) if it looks headerless, reload with header=None + expected names
    if not has_user_date(df):
        df2 = pd.read_csv(path, header=None)

        # CERT-like defaults by file type + column count
        if kind == "http":
            if df2.shape[1] == 5:
                df2.columns = ["id", "date", "user", "pc", "url"]
            elif df2.shape[1] == 6:
                df2.columns = ["id", "date", "user", "pc", "url", "extra"]
            else:
                raise ValueError(f"[{kind}] Unexpected column count in {path}: {df2.shape[1]}")
        elif kind in ("logon", "device"):
            if df2.shape[1] == 5:
                df2.columns = ["id", "date", "user", "pc", "activity"]
            elif df2.shape[1] == 6:
                df2.columns = ["id", "date", "user", "pc", "activity", "extra"]
            else:
                raise ValueError(f"[{kind}] Unexpected column count in {path}: {df2.shape[1]}")
        else:
            raise ValueError(f"[{kind}] Headerless file not supported automatically.")

        df = df2

    # ---- now proceed as before ----
    user_c = _pick_col(df, ["user", "userid", "user_id", "employee", "emp", "subject"])
    time_c = _pick_col(df, ["date", "timestamp", "time", "datetime", "logtime"])
    act_c  = _pick_col(df, ["activity", "action", "event", "status", "result"])
    pc_c   = _pick_col(df, ["pc", "computer", "host", "machine", "device"])
    url_c  = _pick_col(df, ["url", "uri", "request", "domain"])

    if user_c is None or time_c is None:
        raise ValueError(f"[{kind}] Could not detect required columns. Need user+date. Found columns: {list(df.columns)}")

    df["_user"] = df[user_c].astype(str)

    # robust datetime parse (tries default, then dayfirst if needed)
    dt = pd.to_datetime(df[time_c], errors="coerce", utc=False, infer_datetime_format=True)
    if dt.isna().mean() > 0.2:
        dt = pd.to_datetime(df[time_c], errors="coerce", utc=False, infer_datetime_format=True, dayfirst=True)
    df["_dt"] = dt

    df = df.dropna(subset=["_dt"])
    df["_day"] = df["_dt"].dt.date.astype(str)

    df["_act"] = df[act_c].astype(str) if act_c is not None else ""
    df["_pc"]  = df[pc_c].astype(str) if pc_c is not None else ""
    df["_url"] = df[url_c].astype(str) if url_c is not None else ""

    return df

def build_proxy_labels(raw_root: str, label_rate: float):
    logon_path = _find_file(raw_root, "logon.csv")
    device_path = _find_file(raw_root, "device.csv")
    http_path = _find_file(raw_root, "http.csv")

    if not logon_path and not device_path and not http_path:
        raise FileNotFoundError(f"No logon.csv/device.csv/http.csv found under {raw_root}")

    feats = []

    # ---- LOGON FEATURES ----
    if logon_path:
        L = load_log(logon_path, "logon")
        hour = L["_dt"].dt.hour
        L["_offhours"] = ((hour < 6) | (hour >= 20)).astype(int)

        # crude failed detection if we have text
        act = L["_act"].str.lower()
        L["_failed"] = act.str.contains("fail") | act.str.contains("denied") | act.str.contains("error")
        L["_failed"] = L["_failed"].astype(int)

        g = L.groupby(["_user", "_day"])
        f = pd.DataFrame({
            "logon_count": g.size(),
            "logon_offhours": g["_offhours"].sum(),
            "logon_failed": g["_failed"].sum(),
            "logon_unique_pcs": g["_pc"].nunique(),
        }).reset_index()
        feats.append(f)

    # ---- DEVICE FEATURES ----
    if device_path:
        D = load_log(device_path, "device")
        act = D["_act"].str.lower()
        D["_usb_like"] = (
            act.str.contains("connect") |
            act.str.contains("insert") |
            act.str.contains("mount") |
            act.str.contains("usb")
        ).astype(int)

        g = D.groupby(["_user", "_day"])
        f = pd.DataFrame({
            "device_events": g.size(),
            "device_usb_like": g["_usb_like"].sum(),
            "device_unique_pcs": g["_pc"].nunique(),
        }).reset_index()
        feats.append(f)

    # ---- HTTP FEATURES ----
    if http_path:
        H = load_log(http_path, "http")
        # domain extraction
        def domain(u):
            try:
                net = urlparse(u).netloc
                if net:
                    return net.lower()
            except Exception:
                pass
            return ""
        H["_domain"] = H["_url"].map(domain)

        g = H.groupby(["_user", "_day"])
        f = pd.DataFrame({
            "http_events": g.size(),
            "http_unique_domains": g["_domain"].nunique(),
        }).reset_index()
        feats.append(f)

    # ---- MERGE ALL FEATURES ----
    if not feats:
        raise RuntimeError("No features created (files found but parsing failed).")

    df = feats[0]
    for nxt in feats[1:]:
        df = df.merge(nxt, on=["_user", "_day"], how="outer")
    df = df.fillna(0)

    # ---- SCORE (robust, per-user) ----
    metric_cols = [c for c in df.columns if c not in ["_user", "_day"]]
    df["score"] = 0.0
    for c in metric_cols:
        df["score"] += df.groupby("_user")[c].transform(robust_z).clip(lower=0)

    # ---- LABELS (top % by score) ----
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    k = max(1, int(len(df) * label_rate))
    df["label"] = 0
    df.loc[:k-1, "label"] = 1

    labels = df[["_user", "_day", "label", "score"]].rename(columns={"_user": "user", "_day": "date"})
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="data/raw", help="Root folder containing logon.csv/device.csv/http.csv")
    ap.add_argument("--out", default="data/labels.csv", help="Output labels csv path")
    ap.add_argument("--label_rate", type=float, default=0.02, help="Percent of user-days labeled as 1 (e.g., 0.02 = 2%)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = build_proxy_labels(args.raw_root, args.label_rate)
    labels.to_csv(out_path, index=False)

    print(f"✅ Wrote {len(labels)} rows to {out_path}")
    print(labels.head(10).to_string(index=False))

if __name__ == "__main__":
    main()