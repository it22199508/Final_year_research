"""
DEMF Simplified Core
- Single-file core logic for loading CERT-style logs (logon/device/http), feature engineering,
  unsupervised anomaly detection (Autoencoder-like recon via MLPRegressor + OneClassSVM),
  contextual scoring, canary triggers, and report generation.

Design goals:
- No package/import path headaches (no src/ folder). Everything is importable from project root.
- Robust CSV loading: handles BOM headers, weird delimiters, and column-name casing/spaces.
- Works with the dataset schema you shared:
    logon.csv:  id, date, user, pc, activity
    device.csv: id, date, user, pc, activity
    http.csv:   id, date, user, pc, url
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hashlib
import json
import re

import numpy as np
import pandas as pd
import yaml
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor


# -----------------------------
# Config
# -----------------------------

@dataclass
class DEMFConfig:
    # paths
    raw_dir: Path = Path("data/raw")
    model_dir: Path = Path("models/current")
    report_dir: Path = Path("reports/current")

    # privacy
    hash_salt: str = "change_me"

    # detection
    bh_start: int = 8
    bh_end: int = 18
    contamination: float = 0.01
    ae_weight: float = 0.6
    svm_weight: float = 0.4
    context_weight: float = 0.2  # blends contextual score into final

    # canary tokens
    canary_tokens: Tuple[str, ...] = ("DEMF_CANARY_TOKEN", "canary.example.com")

    # modeling
    random_state: int = 42
    max_iter: int = 300

    @staticmethod
    def from_yaml(path: Path) -> "DEMFConfig":
        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = DEMFConfig()

        # safe getters
        data = obj.get("data", {})
        privacy = obj.get("privacy", {})
        detection = obj.get("detection", {})
        canary = obj.get("canary", {})
        modeling = obj.get("modeling", {})

        cfg.raw_dir = Path(data.get("raw_dir", cfg.raw_dir))
        cfg.model_dir = Path(data.get("model_dir", cfg.model_dir))
        cfg.report_dir = Path(data.get("report_dir", cfg.report_dir))

        cfg.hash_salt = str(privacy.get("hash_salt", cfg.hash_salt))

        bh = detection.get("business_hours", {})
        cfg.bh_start = int(bh.get("start", cfg.bh_start))
        cfg.bh_end = int(bh.get("end", cfg.bh_end))
        cfg.contamination = float(detection.get("contamination", cfg.contamination))
        cfg.ae_weight = float(detection.get("ae_weight", cfg.ae_weight))
        cfg.svm_weight = float(detection.get("svm_weight", cfg.svm_weight))
        cfg.context_weight = float(detection.get("context_weight", cfg.context_weight))

        tokens = canary.get("tokens", list(cfg.canary_tokens))
        cfg.canary_tokens = tuple(map(str, tokens))

        cfg.random_state = int(modeling.get("random_state", cfg.random_state))
        cfg.max_iter = int(modeling.get("max_iter", cfg.max_iter))

        return cfg


# -----------------------------
# Robust CSV loading
# -----------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)  # BOM
        .str.strip()
        .str.lower()
    )
    return df


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Read a CSV robustly:
    - utf-8-sig handles BOM
    - if it parses as 1 column (delimiter issue), retry with sep=None (python engine)
    - if the file has *no header row* (common when CSVs are re-saved), detect and re-read with header=None
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    # First pass: assume header exists
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] == 1:
        # delimiter sniff
        df2 = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
        if df2.shape[1] > 1:
            df = df2

    df = _normalize_columns(df)

    # Headerless CERT CSVs: pandas will treat the first data row as column names
    expected_by_file = {
        "http.csv": ["id", "date", "user", "pc", "url"],
        "logon.csv": ["id", "date", "user", "pc", "activity"],
        "device.csv": ["id", "date", "user", "pc", "activity"],
    }
    expected = expected_by_file.get(path.name.lower())
    if expected is not None:
        has_expected_header = set(expected).issubset(set(df.columns))
        if (not has_expected_header) and (df.shape[1] == len(expected)):
            # Re-read correctly (preserves the first record that was previously used as header)
            df2 = pd.read_csv(path, encoding="utf-8-sig", header=None)
            if df2.shape[1] == 1:
                df2 = pd.read_csv(path, encoding="utf-8-sig", header=None, sep=None, engine="python")

            if df2.shape[1] == len(expected):
                df2.columns = expected
                df = _normalize_columns(df2)

    return df



def _ensure_columns(df: pd.DataFrame, needed: List[str], candidates: Dict[str, List[str]], ctx: str) -> pd.DataFrame:
    """
    Ensure canonical columns exist by renaming from candidate synonyms.
    """
    df = df.copy()
    cols = set(df.columns)

    rename_map = {}
    for canonical in needed:
        if canonical in cols:
            continue
        for cand in candidates.get(canonical, []):
            if cand in cols:
                rename_map[cand] = canonical
                cols.add(canonical)
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{ctx}] Missing required columns {missing}. Found columns: {list(df.columns)}"
        )
    return df


def parse_timestamp(s: pd.Series) -> pd.Series:
    """
    Parse timestamp in 'date' column. CERT datasets typically store ISO-like timestamps.
    """
    ts = pd.to_datetime(s, errors="coerce")
    if ts.isna().mean() > 0.2:
        # try common alt formats if many failed
        ts = pd.to_datetime(s, errors="coerce")
    return ts


# -----------------------------
# Loading + standardizing CERT-style logs
# -----------------------------

def load_cert_logs(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load the three core CERT-style CSVs.
    Returns dict with keys: logon, device, http (if file exists).
    """
    out: Dict[str, pd.DataFrame] = {}

    logon_path = raw_dir / "logon.csv"
    device_path = raw_dir / "device.csv"
    http_path = raw_dir / "http.csv"

    if logon_path.exists():
        out["logon"] = read_csv_robust(logon_path)
    if device_path.exists():
        out["device"] = read_csv_robust(device_path)
    if http_path.exists():
        out["http"] = read_csv_robust(http_path)

    if not out:
        raise FileNotFoundError(
            f"No CERT files found in {raw_dir}. Expected at least one of: logon.csv, device.csv, http.csv"
        )
    return out


def standardize_events(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Standardize into a single event table with:
      event_id, timestamp, user, pc, source, action, url(optional)
    """
    parts = []

    candidates = {
        "event_id": ["id", "eventid", "event_id"],
        "timestamp": ["date", "time", "datetime", "timestamp"],
        "user": ["user", "user_id", "userid"],
        "pc": ["pc", "host", "machine", "computer"],
        "action": ["activity", "action", "event", "operation"],
        "url": ["url", "uri", "website"],
    }

    if "logon" in raw:
        df = raw["logon"]
        df = _ensure_columns(df, ["event_id", "timestamp", "user", "pc", "action"], candidates, "logon.csv")
        df["timestamp"] = parse_timestamp(df["timestamp"])
        df["source"] = "logon"
        # normalize action values
        df["action"] = df["action"].astype(str).str.strip().str.lower()
        parts.append(df[["event_id", "timestamp", "user", "pc", "source", "action"]])

    if "device" in raw:
        df = raw["device"]
        df = _ensure_columns(df, ["event_id", "timestamp", "user", "pc", "action"], candidates, "device.csv")
        df["timestamp"] = parse_timestamp(df["timestamp"])
        df["source"] = "device"
        df["action"] = df["action"].astype(str).str.strip().str.lower()
        parts.append(df[["event_id", "timestamp", "user", "pc", "source", "action"]])

    if "http" in raw:
        df = raw["http"]
        df = _ensure_columns(df, ["event_id", "timestamp", "user", "pc", "url"], candidates, "http.csv")
        df["timestamp"] = parse_timestamp(df["timestamp"])
        df["source"] = "http"
        df["action"] = "url_visit"
        parts.append(df[["event_id", "timestamp", "user", "pc", "source", "action", "url"]])

    events = pd.concat(parts, ignore_index=True)

    # Fix missing URL column for non-http sources
    if "url" not in events.columns:
        events["url"] = np.nan
    events["url"] = events["url"].astype("string")

    # De-duplicate / ensure global uniqueness if needed
    events["event_uid"] = events["source"].astype(str) + ":" + events["event_id"].astype(str)

    # drop rows with bad timestamps
    events = events.dropna(subset=["timestamp"]).reset_index(drop=True)

    return events


# -----------------------------
# Privacy: hashing
# -----------------------------

def sha256_hash(value: str, salt: str) -> str:
    h = hashlib.sha256()
    h.update((salt + "|" + value).encode("utf-8"))
    return h.hexdigest()


def apply_privacy(events: pd.DataFrame, salt: str) -> pd.DataFrame:
    """
    Hash usernames, keep original user in a separate column only if you want.
    By default, we keep only user_hash in reports.
    """
    df = events.copy()
    df["user"] = df["user"].astype(str)
    df["user_hash"] = df["user"].apply(lambda x: sha256_hash(x, salt))
    return df


# -----------------------------
# Feature engineering (user-day)
# -----------------------------

def _is_after_hours(ts: pd.Series, bh_start: int, bh_end: int) -> pd.Series:
    hr = ts.dt.hour
    return (hr < bh_start) | (hr >= bh_end)


def _extract_domain(url: pd.Series) -> pd.Series:
    # very lightweight domain parse
    s = url.fillna("").astype(str)
    # strip scheme
    s = s.str.replace(r"^\w+://", "", regex=True)
    # take first path segment
    s = s.str.split("/", n=1).str[0]
    # strip port
    s = s.str.split(":", n=1).str[0]
    s = s.str.lower().str.strip()
    s = s.replace("", np.nan)
    return s


def build_user_day_features(events: pd.DataFrame, bh_start: int, bh_end: int, canary_tokens: Tuple[str, ...]) -> pd.DataFrame:
    """
    Aggregate to user_hash + day.
    Produces a numeric feature table + canary flag.
    """
    df = events.copy()
    df["day"] = df["timestamp"].dt.floor("D")
    df["after_hours"] = _is_after_hours(df["timestamp"], bh_start, bh_end)

    # Canary hit at event-level
    token_pattern = "(?:" + "|".join(re.escape(t) for t in canary_tokens) + ")"
    df["canary_hit_event"] = df["url"].fillna("").astype(str).str.contains(token_pattern, case=False, regex=True)

    # HTTP domain features
    df["domain"] = np.where(df["source"] == "http", _extract_domain(df["url"]), np.nan)

    # logon features
    logon = df[df["source"] == "logon"].copy()
    is_logon = logon["action"].str.contains("logon", na=False)
    logon_logons = logon[is_logon]
    # first/last timestamp of day per user for rough "work duration"
    day_bounds = (
        df.groupby(["user_hash", "day"])["timestamp"]
        .agg(first_ts="min", last_ts="max")
        .reset_index()
    )
    day_bounds["work_duration_hours"] = (day_bounds["last_ts"] - day_bounds["first_ts"]).dt.total_seconds() / 3600.0

    # counts by type
    feat = df.groupby(["user_hash", "day"]).agg(
        total_events=("event_uid", "count"),
        unique_pcs=("pc", "nunique"),
        canary_hit=("canary_hit_event", "max"),
    ).reset_index()

    # logon counts
    if not logon.empty:
        logon_counts = logon_logons.groupby(["user_hash", "day"]).agg(
            logon_count=("event_uid", "count"),
            after_hours_logon_count=("after_hours", "sum"),
        ).reset_index()
    else:
        logon_counts = pd.DataFrame(columns=["user_hash", "day", "logon_count", "after_hours_logon_count"])

    # device counts (USB)
    device = df[df["source"] == "device"].copy()
    is_connect = device["action"].str.contains("connect", na=False)
    device_connects = device[is_connect]
    if not device_connects.empty:
        usb_counts = device_connects.groupby(["user_hash", "day"]).agg(
            usb_connect_count=("event_uid", "count"),
            after_hours_usb_connect_count=("after_hours", "sum"),
        ).reset_index()
    else:
        usb_counts = pd.DataFrame(columns=["user_hash", "day", "usb_connect_count", "after_hours_usb_connect_count"])

    # http counts
    http = df[df["source"] == "http"].copy()
    if not http.empty:
        http_counts = http.groupby(["user_hash", "day"]).agg(
            url_count=("event_uid", "count"),
            after_hours_url_count=("after_hours", "sum"),
            unique_domains=("domain", "nunique"),
        ).reset_index()
    else:
        http_counts = pd.DataFrame(columns=["user_hash", "day", "url_count", "after_hours_url_count", "unique_domains"])

    # merge all
    out = feat.merge(logon_counts, on=["user_hash", "day"], how="left")
    out = out.merge(usb_counts, on=["user_hash", "day"], how="left")
    out = out.merge(http_counts, on=["user_hash", "day"], how="left")
    out = out.merge(day_bounds[["user_hash", "day", "work_duration_hours"]], on=["user_hash", "day"], how="left")

    # fill NAs
    for c in ["logon_count", "after_hours_logon_count", "usb_connect_count", "after_hours_usb_connect_count",
              "url_count", "after_hours_url_count", "unique_domains", "work_duration_hours"]:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    out["canary_hit"] = out["canary_hit"].fillna(False).astype(bool)

    # contextual risk heuristics (0..1)
    out["context_score"] = 0.0
    out.loc[out["after_hours_logon_count"] > 0, "context_score"] += 0.20
    out.loc[out["usb_connect_count"] > 0, "context_score"] += 0.15
    out.loc[out["after_hours_usb_connect_count"] > 0, "context_score"] += 0.35
    out.loc[out["after_hours_url_count"] > 0, "context_score"] += 0.15
    out.loc[out["unique_pcs"] > 1, "context_score"] += 0.15
    out["context_score"] = out["context_score"].clip(0, 1)

    # stable sort
    out = out.sort_values(["day", "user_hash"]).reset_index(drop=True)
    return out



def cfg_to_jsonable(cfg: DEMFConfig) -> dict:
    """
    Convert DEMFConfig into a JSON-serializable dict (Paths -> str, tuples -> lists).
    """
    d = dict(cfg.__dict__)
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
        elif isinstance(v, tuple):
            d[k] = list(v)
    return d

# -----------------------------
# Modeling
# -----------------------------

def _numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"user_hash", "day", "canary_hit"}
    cols = [c for c in df.columns if c not in exclude]
    # keep only numeric + bool
    keep = []
    for c in cols:
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
            keep.append(c)
    return keep


def train_artifacts(user_day: pd.DataFrame, cfg: DEMFConfig) -> Dict[str, object]:
    """
    Train:
      - StandardScaler
      - MLPRegressor "autoencoder"
      - OneClassSVM
    Fit on early portion of timeline (train split by day).
    """
    df = user_day.copy().sort_values("day")
    days = df["day"].sort_values().unique()
    if len(days) < 10:
        raise ValueError("Not enough days to train. Need at least ~10 distinct days.")

    split_idx = int(len(days) * 0.7)
    train_days = set(days[:split_idx])

    train_df = df[df["day"].isin(train_days)].reset_index(drop=True)

    feature_cols = _numeric_feature_cols(df)
    X_train = train_df[feature_cols].astype(float).to_numpy()
    X_all = df[feature_cols].astype(float).to_numpy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_all_s = scaler.transform(X_all)

    # MLPRegressor used as an "autoencoder": train to reconstruct inputs
    input_dim = X_train_s.shape[1]
    h1 = max(8, input_dim // 2)
    bottleneck = max(3, input_dim // 4)
    hidden = (h1, bottleneck, h1)

    ae = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        early_stopping=True,
        n_iter_no_change=10,
        verbose=False,
    )
    ae.fit(X_train_s, X_train_s)

    # One-class SVM on scaled features
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=min(0.5, max(0.001, cfg.contamination * 2)))
    svm.fit(X_train_s)

    # threshold from train combined score
    scores_all = score_user_days(df, scaler, ae, svm, feature_cols, cfg, fit_threshold_on_train=True, train_mask=df["day"].isin(train_days))
    threshold = float(scores_all["threshold"].iloc[0])

    artifacts = {
        "scaler": scaler,
        "autoencoder": ae,
        "ocsvm": svm,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "train_days": [str(d) for d in sorted(train_days)],
        "config_snapshot": cfg_to_jsonable(cfg),
    }
    return artifacts


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.all(np.isfinite(x)) is False:
        x = np.nan_to_num(x, nan=np.nanmedian(x), posinf=np.nanmax(x[np.isfinite(x)]), neginf=np.nanmin(x[np.isfinite(x)]))
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def score_user_days(
    user_day: pd.DataFrame,
    scaler: StandardScaler,
    autoencoder: MLPRegressor,
    ocsvm: OneClassSVM,
    feature_cols: List[str],
    cfg: DEMFConfig,
    fit_threshold_on_train: bool = False,
    train_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Score each user-day with:
      - recon_error (autoencoder MSE)
      - svm_anom (negative decision_function)
      - combined_score
      - final_score (combined blended with context)
      - alert boolean with threshold
    """
    df = user_day.copy().reset_index(drop=True)

    X = df[feature_cols].astype(float).to_numpy()
    Xs = scaler.transform(X)

    X_hat = autoencoder.predict(Xs)
    recon_error = np.mean((Xs - X_hat) ** 2, axis=1)

    # decision_function: larger => more normal. Convert to anomaly with negative sign.
    svm_normal = ocsvm.decision_function(Xs).ravel()
    svm_anom = -svm_normal

    recon_n = _minmax_norm(recon_error)
    svm_n = _minmax_norm(svm_anom)

    combined = cfg.ae_weight * recon_n + cfg.svm_weight * svm_n
    combined = _minmax_norm(combined)

    final = (1 - cfg.context_weight) * combined + cfg.context_weight * df["context_score"].to_numpy()
    final = _minmax_norm(final)

    # threshold
    if fit_threshold_on_train:
        if train_mask is None:
            raise ValueError("train_mask required when fit_threshold_on_train=True")
        train_final = final[np.asarray(train_mask)]
        threshold = float(np.quantile(train_final, 1 - cfg.contamination))
    else:
        # placeholder; caller will overwrite from saved artifact
        threshold = float(np.quantile(final, 1 - cfg.contamination))

    alert = (final >= threshold) | (df.get("canary_hit", False).astype(bool).to_numpy())

    # severity
    severity = np.where(df.get("canary_hit", False).astype(bool).to_numpy(), "critical",
                        np.where(final >= max(threshold, 0.95), "high",
                                 np.where(final >= max(threshold, 0.85), "medium", "low")))

    # explainability: top deviations vs median in scaled space
    med = np.median(Xs, axis=0)
    mad = np.median(np.abs(Xs - med), axis=0) + 1e-6
    dev = np.abs((Xs - med) / mad)
    top_idx = np.argsort(-dev, axis=1)[:, :3]
    top_feats = []
    for i in range(len(df)):
        feats = [feature_cols[j] for j in top_idx[i]]
        top_feats.append(", ".join(feats))

    df_out = df[["user_hash", "day"]].copy()
    df_out["recon_error"] = recon_error
    df_out["svm_anom"] = svm_anom
    df_out["combined_score"] = combined
    df_out["final_score"] = final
    df_out["threshold"] = threshold
    df_out["alert"] = alert
    df_out["severity"] = severity
    df_out["top_factors"] = top_feats

    # Analyst-friendly explanation
    expl = []
    for i in range(len(df_out)):
        parts = []
        if bool(df.get("canary_hit", False).iloc[i]) if "canary_hit" in df.columns else False:
            parts.append("Canary token matched")
        if df.loc[i, "after_hours_usb_connect_count"] > 0:
            parts.append("After-hours USB usage")
        if df.loc[i, "after_hours_logon_count"] > 0:
            parts.append("After-hours logon")
        if df.loc[i, "after_hours_url_count"] > 0:
            parts.append("After-hours web activity")
        if df.loc[i, "unique_pcs"] > 1:
            parts.append("Multiple PCs used in a day")
        if not parts:
            parts.append("Statistical anomaly vs baseline")
        parts.append(f"Top factors: {df_out.loc[i, 'top_factors']}")
        expl.append(" | ".join(parts))
    df_out["explanation"] = expl

    return df_out


# -----------------------------
# Save / load artifacts
# -----------------------------

def save_artifacts(artifacts: Dict[str, object], model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    dump(artifacts["scaler"], model_dir / "scaler.joblib")
    dump(artifacts["autoencoder"], model_dir / "autoencoder.joblib")
    dump(artifacts["ocsvm"], model_dir / "ocsvm.joblib")
    (model_dir / "feature_cols.json").write_text(json.dumps(artifacts["feature_cols"], indent=2), encoding="utf-8")
    meta = {
        "threshold": artifacts["threshold"],
        "train_days": artifacts["train_days"],
        "config_snapshot": artifacts["config_snapshot"],
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_artifacts(model_dir: Path) -> Dict[str, object]:
    scaler = load(model_dir / "scaler.joblib")
    autoencoder = load(model_dir / "autoencoder.joblib")
    ocsvm = load(model_dir / "ocsvm.joblib")
    feature_cols = json.loads((model_dir / "feature_cols.json").read_text(encoding="utf-8"))
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    return {
        "scaler": scaler,
        "autoencoder": autoencoder,
        "ocsvm": ocsvm,
        "feature_cols": feature_cols,
        "threshold": float(meta["threshold"]),
        "meta": meta,
    }


# -----------------------------
# End-to-end pipeline helpers
# -----------------------------

def run_train_and_detect(cfg: DEMFConfig) -> Dict[str, pd.DataFrame]:
    """
    Full pipeline:
      - load raw logs
      - standardize events
      - hash users
      - build user-day features
      - train models
      - score all user-days
      - save artifacts + reports
    """
    raw = load_cert_logs(cfg.raw_dir)
    events = standardize_events(raw)
    events = apply_privacy(events, cfg.hash_salt)

    user_day = build_user_day_features(events, cfg.bh_start, cfg.bh_end, cfg.canary_tokens)

    artifacts = train_artifacts(user_day, cfg)
    save_artifacts(artifacts, cfg.model_dir)

    # score using saved threshold
    loaded = load_artifacts(cfg.model_dir)
    scores = score_user_days(
        user_day,
        loaded["scaler"],
        loaded["autoencoder"],
        loaded["ocsvm"],
        loaded["feature_cols"],
        cfg,
        fit_threshold_on_train=False,
    )
    scores["threshold"] = loaded["threshold"]
    scores["alert"] = (scores["final_score"] >= loaded["threshold"]) | (user_day["canary_hit"].to_numpy())
    scores["severity"] = np.where(user_day["canary_hit"].to_numpy(), "critical",
                                  np.where(scores["final_score"] >= max(loaded["threshold"], 0.95), "high",
                                           np.where(scores["final_score"] >= max(loaded["threshold"], 0.85), "medium", "low")))

    # save reports
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(cfg.report_dir / "alerts.csv", index=False)
    user_day.to_csv(cfg.report_dir / "features.csv", index=False)
    events_to_save = events.drop(columns=["user"], errors="ignore")
    events_to_save.to_csv(cfg.report_dir / "events_standardized.csv", index=False)

    return {"events": events, "features": user_day, "scores": scores}


def run_detect_only(cfg: DEMFConfig) -> Dict[str, pd.DataFrame]:
    """
    Detect-only pipeline:
      - load raw logs
      - standardize + hash
      - features
      - load artifacts from cfg.model_dir
      - score
      - save reports
    """
    raw = load_cert_logs(cfg.raw_dir)
    events = standardize_events(raw)
    events = apply_privacy(events, cfg.hash_salt)
    user_day = build_user_day_features(events, cfg.bh_start, cfg.bh_end, cfg.canary_tokens)

    loaded = load_artifacts(cfg.model_dir)
    scores = score_user_days(
        user_day,
        loaded["scaler"],
        loaded["autoencoder"],
        loaded["ocsvm"],
        loaded["feature_cols"],
        cfg,
        fit_threshold_on_train=False,
    )
    scores["threshold"] = loaded["threshold"]
    scores["alert"] = (scores["final_score"] >= loaded["threshold"]) | (user_day["canary_hit"].to_numpy())

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(cfg.report_dir / "alerts.csv", index=False)
    user_day.to_csv(cfg.report_dir / "features.csv", index=False)
    events_to_save = events.drop(columns=["user"], errors="ignore")
    events_to_save.to_csv(cfg.report_dir / "events_standardized.csv", index=False)

    return {"events": events, "features": user_day, "scores": scores}
