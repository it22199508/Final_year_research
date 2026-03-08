"""Streamlit monitoring dashboard for DEMF (release-style).

This dashboard is designed for *local analyst use* (SOC/blue team prototype).
It visualizes:
  - user-day anomaly scores (Autoencoder-like + One-Class SVM)
  - contextual risk factors (after-hours, USB, multi-PC)
  - canary confirmations (high confidence)

Inputs
  - CERT-style logs in data/raw (or configured path): logon.csv, device.csv, http.csv

Outputs (produced by the pipeline)
  - reports/<name>/alerts.csv
  - reports/<name>/features.csv
  - reports/<name>/events_standardized.csv

Run
  streamlit run app_streamlit.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joblib import load as joblib_load

from demf_core import (
    DEMFConfig,
    EVAL_BY_DAY_FILE,
    EVAL_SCORED_FILE,
    EVAL_SUMMARY_FILE,
    EVAL_THRESHOLD_FILE,
    run_detect_only,
    run_train_and_detect,
)


# ----------------------------
# Page config + minimal styling
# ----------------------------

st.set_page_config(
    page_title="DEMF Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
  .small-muted { color: rgba(0,0,0,0.6); font-size: 0.92rem; }
  .pill { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.85rem; }
  .pill-critical { background: rgba(220,20,60,0.12); color: rgb(220,20,60); border: 1px solid rgba(220,20,60,0.3); }
  .pill-high { background: rgba(255,140,0,0.12); color: rgb(255,140,0); border: 1px solid rgba(255,140,0,0.3); }
  .pill-medium { background: rgba(30,144,255,0.10); color: rgb(30,144,255); border: 1px solid rgba(30,144,255,0.25); }
  .pill-low { background: rgba(0,128,0,0.10); color: rgb(0,128,0); border: 1px solid rgba(0,128,0,0.25); }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Helpers
# ----------------------------

@st.cache_data(show_spinner=False)
def _read_csv(path: Path, parse_dates: Optional[list] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, parse_dates=parse_dates)


@st.cache_resource(show_spinner=False)
def _load_scaler(model_dir: Path):
    p = model_dir / "scaler.joblib"
    if not p.exists():
        return None
    return joblib_load(p)


@st.cache_data(show_spinner=False)
def _load_feature_cols(model_dir: Path) -> Optional[list]:
    p = model_dir / "feature_cols.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _domain(url: str) -> Optional[str]:
    if url is None or (isinstance(url, float) and np.isnan(url)):
        return None
    s = str(url).strip().lower()
    if not s:
        return None
    # strip scheme
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0]
    s = s.split(":", 1)[0]
    return s or None


def _severity_pill(sev: str) -> str:
    sev = (sev or "").lower()
    cls = {
        "critical": "pill-critical",
        "high": "pill-high",
        "medium": "pill-medium",
        "low": "pill-low",
    }.get(sev, "pill-low")
    return f"<span class='pill {cls}'>{sev.upper()}</span>"


def _safe_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_datetime(df[col], errors="coerce")
    return s


@st.cache_data(show_spinner=False)
def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_read_csv(path: Path, parse_dates: Optional[list] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _read_csv(path, parse_dates=parse_dates)


@st.cache_data(show_spinner=False)
def _load_optional_evaluation(report_dir: Path) -> Optional[dict]:
    summary_path = report_dir / EVAL_SUMMARY_FILE
    if not summary_path.exists():
        return None

    summary = _read_json(summary_path)
    by_day = _maybe_read_csv(report_dir / EVAL_BY_DAY_FILE, parse_dates=["day"])
    threshold_curve = _maybe_read_csv(report_dir / EVAL_THRESHOLD_FILE)
    scored = _maybe_read_csv(report_dir / EVAL_SCORED_FILE, parse_dates=["day"])

    return {
        "summary": summary,
        "by_day": by_day,
        "threshold_curve": threshold_curve,
        "scored": scored,
    }


def _eval_threshold_curve(curve: pd.DataFrame, current_threshold: Optional[float], best_threshold: Optional[float]) -> go.Figure:
    fig = go.Figure()
    if curve.empty:
        fig.update_layout(title="F1 / precision / recall vs threshold", height=360)
        return fig

    for metric, label in [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]:
        if metric in curve.columns:
            fig.add_trace(go.Scatter(x=curve["threshold"], y=curve[metric], mode="lines", name=label))

    if current_threshold is not None:
        fig.add_vline(x=float(current_threshold), line_dash="dash", annotation_text="current")
    if best_threshold is not None:
        fig.add_vline(x=float(best_threshold), line_dash="dot", annotation_text="best F1")

    fig.update_layout(
        title="F1 / precision / recall vs threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def _eval_daily_scores(by_day: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if by_day.empty:
        fig.update_layout(title="Daily evaluation scores", height=320)
        return fig

    for metric, label in [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]:
        if metric in by_day.columns:
            fig.add_trace(go.Scatter(x=by_day["day"], y=by_day[metric], mode="lines+markers", name=label))

    fig.update_layout(
        title="Daily evaluation scores",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def _confusion_matrix_fig(summary: dict) -> go.Figure:
    z = [
        [int(summary.get("tn", 0)), int(summary.get("fp", 0))],
        [int(summary.get("fn", 0)), int(summary.get("tp", 0))],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Pred 0", "Pred 1"],
            y=["Actual 0", "Actual 1"],
            text=z,
            texttemplate="%{text}",
            hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title="Confusion matrix", height=360, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _kpi_row(scores: pd.DataFrame, events: Optional[pd.DataFrame]) -> None:
    total_user_days = int(len(scores))
    total_alerts = int(scores["alert"].sum()) if "alert" in scores.columns else 0
    sev_counts = scores["severity"].value_counts(dropna=False).to_dict() if "severity" in scores.columns else {}

    total_events = int(len(events)) if events is not None else 0
    unique_users = int(scores["user_hash"].nunique()) if "user_hash" in scores.columns else 0
    days = int(scores["day"].nunique()) if "day" in scores.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Users", f"{unique_users:,}")
    c2.metric("Days", f"{days:,}")
    c3.metric("User-days", f"{total_user_days:,}")
    c4.metric("Events", f"{total_events:,}")
    c5.metric("Alerts", f"{total_alerts:,}")
    c6.metric("Critical", f"{sev_counts.get('critical', 0):,}")


def _prepare_join(scores: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    s = scores.copy()
    f = feats.copy()
    s["day"] = _safe_datetime(s, "day")
    f["day"] = _safe_datetime(f, "day")
    out = s.merge(f, on=["user_hash", "day"], how="left", suffixes=("", "_feat"))
    return out


def _alerts_over_time(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()
    df["day"] = _safe_datetime(df, "day").dt.floor("D")
    df = df[df["alert"] == True] if "alert" in df.columns else df
    if df.empty:
        return pd.DataFrame(columns=["day", "severity", "count"])
    g = df.groupby(["day", "severity"]).size().reset_index(name="count")
    return g


def _events_over_time(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df["timestamp"] = _safe_datetime(df, "timestamp")
    df["day"] = df["timestamp"].dt.floor("D")
    g = df.groupby(["day", "source"]).size().reset_index(name="count")
    return g


def _risk_ranking(scores: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    df = _prepare_join(scores, feats)
    # Risk score: emphasizes alerts and canary confirmations
    df["risk"] = df["final_score"].fillna(0.0)
    if "canary_hit" in df.columns:
        df.loc[df["canary_hit"].astype(bool), "risk"] = np.maximum(df.loc[df["canary_hit"].astype(bool), "risk"], 1.0)
    agg = df.groupby("user_hash").agg(
        risk_sum=("risk", "sum"),
        risk_mean=("risk", "mean"),
        alerts=("alert", "sum"),
        critical=("severity", lambda s: int((s == "critical").sum()) if s is not None else 0),
        days=("day", "nunique"),
    ).reset_index()
    agg = agg.sort_values(["risk_sum", "alerts"], ascending=False)
    return agg


def _score_hist(scores: pd.DataFrame, threshold: Optional[float]) -> go.Figure:
    df = scores.copy()
    fig = px.histogram(df, x="final_score", nbins=40, title="Final score distribution")
    if threshold is not None:
        fig.add_vline(x=float(threshold), line_dash="dash", annotation_text=f"threshold={threshold:.3f}")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _stacked_alerts(g: pd.DataFrame) -> go.Figure:
    if g.empty:
        return go.Figure().update_layout(title="Alerts over time", height=320)
    fig = px.area(g, x="day", y="count", color="severity", title="Alerts over time (stacked by severity)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _events_mix(g: pd.DataFrame) -> go.Figure:
    if g.empty:
        return go.Figure().update_layout(title="Event volume over time", height=320)
    fig = px.line(g, x="day", y="count", color="source", title="Event volume over time")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _scatter_diag(scores: pd.DataFrame) -> go.Figure:
    df = scores.copy()
    fig = px.scatter(
        df,
        x="recon_error",
        y="svm_anom",
        color="severity" if "severity" in df.columns else None,
        hover_data=["user_hash", "day", "final_score", "alert"],
        title="Model diagnostics: reconstruction error vs SVM anomaly",
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _feature_deviation_row(row: pd.Series, scaler, feature_cols: list) -> pd.DataFrame:
    """Compute |z|-like deviation using scaler mean/scale."""
    if scaler is None or feature_cols is None:
        return pd.DataFrame(columns=["feature", "value", "z_abs"])

    vals = row.reindex(feature_cols).astype(float).to_numpy()
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        return pd.DataFrame(columns=["feature", "value", "z_abs"])

    z = (vals - mean) / (scale + 1e-9)
    out = pd.DataFrame({"feature": feature_cols, "value": vals, "z_abs": np.abs(z)})
    out = out.sort_values("z_abs", ascending=False)
    return out


def _hourly_heatmap(events: pd.DataFrame, user_hash: str, day: pd.Timestamp) -> go.Figure:
    df = events.copy()
    df["timestamp"] = _safe_datetime(df, "timestamp")
    df = df[df["user_hash"] == user_hash]
    df = df[df["timestamp"].dt.floor("D") == day.floor("D")]
    if df.empty:
        return go.Figure().update_layout(title="Hourly activity heatmap", height=260)

    df["hour"] = df["timestamp"].dt.hour
    piv = df.groupby(["source", "hour"]).size().reset_index(name="count")
    piv = piv.pivot(index="source", columns="hour", values="count").fillna(0)

    fig = px.imshow(piv, aspect="auto", title="Hourly activity (by source)")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=45, b=10))
    return fig


# ----------------------------
# Sidebar: configuration + actions
# ----------------------------

st.title("🛡️ DEMF Monitoring Dashboard")
st.markdown(
    "<div class='small-muted'>Platform-aware, privacy-preserving anomaly monitoring over user activity logs (CERT-style). "
    "Autoencoder-like reconstruction + One-Class SVM + contextual scoring + canary confirmation.</div>",
    unsafe_allow_html=True,
)

st.sidebar.header("Run settings")

config_path = st.sidebar.text_input("config.yaml", value="config.yaml")
if Path(config_path).exists():
    cfg = DEMFConfig.from_yaml(Path(config_path))
else:
    cfg = DEMFConfig()

cfg.raw_dir = Path(st.sidebar.text_input("Raw data folder", value=str(cfg.raw_dir)))
cfg.model_dir = Path(st.sidebar.text_input("Model folder", value=str(cfg.model_dir)))
cfg.report_dir = Path(st.sidebar.text_input("Reports folder", value=str(cfg.report_dir)))
labels_default = str(cfg.labels_path) if cfg.labels_path is not None else ""
labels_input = st.sidebar.text_input("Labels CSV (optional)", value=labels_default, help="Used to compute precision / recall / F1 in the dashboard.")
cfg.labels_path = Path(labels_input) if labels_input.strip() else None

st.sidebar.subheader("Business hours")
cfg.bh_start = st.sidebar.number_input("Start hour", 0, 23, int(cfg.bh_start))
cfg.bh_end = st.sidebar.number_input("End hour", 1, 24, int(cfg.bh_end))

st.sidebar.subheader("Detection")
cfg.contamination = st.sidebar.slider("Expected anomaly rate", 0.001, 0.10, float(cfg.contamination), step=0.001)
cfg.ae_weight = st.sidebar.slider("Reconstruction weight", 0.0, 1.0, float(cfg.ae_weight), step=0.05)
cfg.svm_weight = st.sidebar.slider("SVM weight", 0.0, 1.0, float(cfg.svm_weight), step=0.05)
cfg.context_weight = st.sidebar.slider("Context blend", 0.0, 0.5, float(cfg.context_weight), step=0.05)

st.sidebar.subheader("Privacy")
cfg.hash_salt = st.sidebar.text_input("SHA-256 salt", value=str(cfg.hash_salt))

st.sidebar.subheader("Canary tokens")
tokens_txt = st.sidebar.text_area("One token per line", value="\n".join(cfg.canary_tokens))
cfg.canary_tokens = tuple([t.strip() for t in tokens_txt.splitlines() if t.strip()])

st.sidebar.divider()

cA, cB, cC = st.sidebar.columns(3)
btn_train = cA.button("Train", use_container_width=True)
btn_detect = cB.button("Detect", use_container_width=True)
btn_load = cC.button("Load", use_container_width=True)

st.sidebar.caption("Tip: Use **Train** the first time. Then use **Detect** for new data.")


# ----------------------------
# Run / Load
# ----------------------------

out = None
err: Optional[str] = None

if btn_train:
    with st.spinner("Training models and running detection..."):
        try:
            out = run_train_and_detect(cfg)
        except Exception as e:
            err = str(e)
elif btn_detect:
    with st.spinner("Running detection with existing models..."):
        try:
            out = run_detect_only(cfg)
        except Exception as e:
            err = str(e)

if err:
    st.error(err)
    st.stop()

alerts_path = cfg.report_dir / "alerts.csv"
features_path = cfg.report_dir / "features.csv"
events_path = cfg.report_dir / "events_standardized.csv"

if out is None:
    if not (alerts_path.exists() and features_path.exists() and events_path.exists()):
        st.info("No reports found yet. Click **Train** (first run) or **Detect** (requires existing models).")
        st.stop()
    scores = _read_csv(alerts_path, parse_dates=["day"])
    feats = _read_csv(features_path, parse_dates=["day"])
    events = _read_csv(events_path, parse_dates=["timestamp"])
    evaluation = _load_optional_evaluation(cfg.report_dir)
else:
    scores = out["scores"].copy()
    feats = out["features"].copy()
    events = out["events"].copy()
    evaluation = None
    if "evaluation_summary" in out:
        evaluation = {
            "summary": out["evaluation_summary"],
            "by_day": out.get("evaluation_by_day", pd.DataFrame()).copy(),
            "threshold_curve": out.get("evaluation_threshold_curve", pd.DataFrame()).copy(),
            "scored": out.get("evaluation_scored", pd.DataFrame()).copy(),
        }

# Normalize datetime
a = scores.copy()
a["day"] = _safe_datetime(a, "day")
scores = a
f = feats.copy()
f["day"] = _safe_datetime(f, "day")
feats = f
e = events.copy()
e["timestamp"] = _safe_datetime(e, "timestamp")
events = e

# Compute model threshold if present
threshold = None
if "threshold" in scores.columns and scores["threshold"].notna().any():
    threshold = float(scores["threshold"].dropna().iloc[0])

if evaluation is not None:
    if not evaluation.get("by_day", pd.DataFrame()).empty and "day" in evaluation["by_day"].columns:
        tmp_eval = evaluation["by_day"].copy()
        tmp_eval["day"] = pd.to_datetime(tmp_eval["day"], errors="coerce")
        evaluation["by_day"] = tmp_eval
    if not evaluation.get("scored", pd.DataFrame()).empty and "day" in evaluation["scored"].columns:
        tmp_scored = evaluation["scored"].copy()
        tmp_scored["day"] = pd.to_datetime(tmp_scored["day"], errors="coerce")
        evaluation["scored"] = tmp_scored


# ----------------------------
# Data health strip
# ----------------------------

with st.expander("Data health & ingestion", expanded=False):
    st.write("**Raw folder:**", str(cfg.raw_dir.resolve()))
    st.write("**Reports folder:**", str(cfg.report_dir.resolve()))
    st.write("**Model folder:**", str(cfg.model_dir.resolve()))

    req_files = ["logon.csv", "device.csv", "http.csv"]
    cols = st.columns(3)
    for i, fname in enumerate(req_files):
        p = cfg.raw_dir / fname
        ok = p.exists()
        cols[i].metric(fname, "FOUND" if ok else "MISSING")

    labels_found = cfg.labels_path.exists() if cfg.labels_path is not None else False
    st.write("**Labels file:**", str(cfg.labels_path) if cfg.labels_path is not None else "Not provided")
    st.write("**Evaluation status:**", "READY" if evaluation is not None else ("LABELS FOUND - run Detect/Train" if labels_found else "NO LABELS"))

    if not events.empty:
        st.markdown("**Event mix**")
        mix = events["source"].value_counts().rename_axis("source").reset_index(name="events")
        st.dataframe(mix, use_container_width=True, hide_index=True)


# ----------------------------
# Main layout
# ----------------------------

_kpi_row(scores, events)

joined = _prepare_join(scores, feats)

# Tabs
overview_tab, alerts_tab, user_tab, analytics_tab, eval_tab, model_tab = st.tabs(
    ["Overview", "Alert Explorer", "User Investigation", "Behavior Analytics", "Evaluation / F1", "Model & Config"]
)


# ----------------------------
# Overview
# ----------------------------
with overview_tab:
    if evaluation is not None:
        summary = evaluation["summary"]
        st.markdown("#### Ground-truth evaluation")
        if summary.get("status") == "no_overlap":
            st.warning("A labels file was found, but it did not overlap with the scored user-day records. Check the hashing salt, user identifier, and day/date columns.")
        else:
            ec1, ec2, ec3, ec4, ec5 = st.columns(5)
            ec1.metric("F1", f"{float(summary.get('f1', 0.0)):.3f}", delta=f"best {float(summary.get('best_f1', 0.0)):.3f}" if summary.get("best_f1") is not None else None)
            ec2.metric("Precision", f"{float(summary.get('precision', 0.0)):.3f}")
            ec3.metric("Recall", f"{float(summary.get('recall', 0.0)):.3f}")
            ec4.metric("Labeled rows", f"{int(summary.get('labeled_rows', 0)):,}")
            ec5.metric("Positive labels", f"{int(summary.get('positive_labels', 0)):,}")
            st.caption(f"Labels file: {summary.get('labels_path', '')}")

    left, right = st.columns([1.15, 0.85])

    g_alerts = _alerts_over_time(scores)
    g_events = _events_over_time(events) if events is not None else pd.DataFrame()

    with left:
        st.plotly_chart(_stacked_alerts(g_alerts), use_container_width=True)

    with right:
        st.plotly_chart(_score_hist(scores, threshold), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_events_mix(g_events), use_container_width=True)

    with c2:
        # After-hours ratio over time
        if {"after_hours_logon_count", "logon_count"}.issubset(set(joined.columns)):
            tmp = joined.copy()
            tmp["day"] = tmp["day"].dt.floor("D")
            tmp["after_hours_ratio"] = (tmp["after_hours_logon_count"] / (tmp["logon_count"] + 1e-6)).clip(0, 1)
            d = tmp.groupby("day")["after_hours_ratio"].mean().reset_index()
            fig = px.line(d, x="day", y="after_hours_ratio", title="After-hours logon ratio (mean across users)")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("After-hours ratio chart unavailable (missing logon features).")


# ----------------------------
# Alert Explorer
# ----------------------------
with alerts_tab:
    st.subheader("Triage queue")

    min_day = scores["day"].min()
    max_day = scores["day"].max()

    flt1, flt2, flt3, flt4 = st.columns([1.4, 1.2, 1.2, 2.2])
    date_range = flt1.date_input("Date range", value=(min_day.date(), max_day.date()))

    sev_options = sorted(scores["severity"].dropna().unique().tolist()) if "severity" in scores.columns else []
    sev_default = sev_options if sev_options else None
    sev_sel = flt2.multiselect("Severity", options=sev_options, default=sev_default)

    min_score = flt3.slider("Min final score", 0.0, 1.0, 0.0, step=0.01)
    user_search = flt4.text_input("User hash contains", value="")

    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

    view = joined[(joined["day"] >= start_dt) & (joined["day"] < end_dt)].copy()
    if sev_sel:
        view = view[view["severity"].isin(sev_sel)].copy()
    view = view[view["final_score"] >= float(min_score)].copy()
    if user_search.strip():
        view = view[view["user_hash"].str.contains(user_search.strip(), case=False, na=False)].copy()

    # Sort by alert first, then score
    if "alert" in view.columns:
        view = view.sort_values(["alert", "final_score"], ascending=[False, False])
    else:
        view = view.sort_values(["final_score"], ascending=[False])

    # Charts: daily alerts + severity
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        daily = view[view["alert"] == True].groupby(view["day"].dt.floor("D")).size().reset_index(name="alerts")
        fig = px.bar(daily, x="day", y="alerts", title="Alerts per day (filtered)")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sev = view[view["alert"] == True]["severity"].value_counts().reset_index()
        sev.columns = ["severity", "count"]
        fig = px.pie(sev, names="severity", values="count", title="Severity split (filtered)")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Alerts table")
    display_cols = [
        "day",
        "user_hash",
        "final_score",
        "severity",
        "alert",
        "explanation",
        "top_factors",
        "canary_hit",
        "after_hours_usb_connect_count",
        "after_hours_logon_count",
        "unique_pcs",
    ]
    display_cols = [c for c in display_cols if c in view.columns]

    st.dataframe(view[display_cols].head(500), use_container_width=True, hide_index=True)

    st.download_button(
        "Download filtered CSV",
        data=view[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="demf_alerts_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ----------------------------
# User Investigation
# ----------------------------
with user_tab:
    st.subheader("Investigate a user")

    users = sorted(scores["user_hash"].unique().tolist())
    u = st.selectbox("User hash", options=users, index=0 if users else None)
    if not u:
        st.info("No users available.")
        st.stop()

    u_df = joined[joined["user_hash"] == u].sort_values("day")

    # Summary KPIs
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Days observed", int(u_df["day"].nunique()))
    s2.metric("Alerts", int(u_df["alert"].sum()))
    s3.metric("Max score", f"{u_df['final_score'].max():.3f}")
    s4.metric("Avg score", f"{u_df['final_score'].mean():.3f}")
    s5.metric("Canary hits", int(u_df.get("canary_hit", False).astype(bool).sum()))

    # Score trend with threshold
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_df["day"], y=u_df["final_score"], mode="lines+markers", name="final_score"))
    if threshold is not None:
        fig.add_trace(go.Scatter(x=u_df["day"], y=[threshold] * len(u_df), mode="lines", name="threshold"))
    fig.update_layout(title="User score trend", height=330, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Most suspicious days
    st.markdown("#### Top suspicious days")
    top_days = u_df.sort_values(["alert", "final_score"], ascending=[False, False]).head(30)
    cols = ["day", "final_score", "severity", "alert", "explanation", "top_factors"]
    cols = [c for c in cols if c in top_days.columns]
    st.dataframe(top_days[cols], use_container_width=True, hide_index=True)

    # Day drilldown
    st.markdown("#### Day drilldown")
    day_options = u_df["day"].dt.floor("D").dropna().unique()
    day_sel = st.selectbox("Select a day", options=sorted(day_options), format_func=lambda d: str(pd.to_datetime(d).date()))
    day_sel = pd.to_datetime(day_sel).floor("D")

    row = u_df[u_df["day"].dt.floor("D") == day_sel].sort_values("final_score", ascending=False).head(1)
    if row.empty:
        st.info("No record for that day.")
    else:
        r = row.iloc[0]
        st.markdown(
            f"**Severity:** {_severity_pill(str(r.get('severity','low')))} &nbsp; "
            f"**final_score:** `{float(r['final_score']):.3f}` &nbsp; "
            f"**top_factors:** `{r.get('top_factors','')}`",
            unsafe_allow_html=True,
        )
        st.caption(r.get("explanation", ""))

        # Feature deviations
        scaler = _load_scaler(cfg.model_dir)
        feature_cols = _load_feature_cols(cfg.model_dir)
        dev = _feature_deviation_row(r, scaler, feature_cols) if feature_cols else pd.DataFrame()

        c1, c2 = st.columns([0.55, 0.45])
        with c1:
            if not dev.empty:
                fig = px.bar(dev.head(12), x="z_abs", y="feature", orientation="h", title="Top deviating features (|z|)")
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature deviation chart unavailable (missing scaler/feature_cols in model folder).")

        with c2:
            # Hourly heatmap
            if "user_hash" in events.columns:
                st.plotly_chart(_hourly_heatmap(events, u, day_sel), use_container_width=True)
            else:
                st.info("Event heatmap unavailable (events missing user_hash).")

        # Event details for the day
        st.markdown("#### Events for selected day")
        ev = events.copy()
        ev["timestamp"] = _safe_datetime(ev, "timestamp")
        ev = ev[(ev["user_hash"] == u) & (ev["timestamp"].dt.floor("D") == day_sel)]
        if not ev.empty:
            ev = ev.sort_values("timestamp")
            ev["domain"] = ev.get("url", pd.Series([None] * len(ev))).apply(_domain)
            st.dataframe(
                ev[[c for c in ["timestamp", "source", "action", "pc", "domain", "url"] if c in ev.columns]].head(1000),
                use_container_width=True,
                hide_index=True,
            )

            # Domain leaderboard
            http = ev[ev["source"] == "http"].copy()
            if not http.empty:
                topd = http["domain"].value_counts().head(15).reset_index()
                topd.columns = ["domain", "visits"]
                fig = px.bar(topd, x="visits", y="domain", orientation="h", title="Top domains (that day)")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No events for that day.")


# ----------------------------
# Behavior Analytics
# ----------------------------
with analytics_tab:
    st.subheader("Population analytics")

    rank = _risk_ranking(scores, feats)
    topn = st.slider("Top users", 5, 100, 25)

    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        fig = px.bar(rank.head(topn), x="risk_sum", y="user_hash", orientation="h", title="Top risky users (risk_sum)")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Risk leaderboard")
        st.dataframe(rank.head(topn), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Scatter: after-hours USB vs score
    if {"after_hours_usb_connect_count", "usb_connect_count", "final_score"}.issubset(set(joined.columns)):
        tmp = joined.copy()
        fig = px.scatter(
            tmp,
            x="after_hours_usb_connect_count",
            y="final_score",
            color="severity" if "severity" in tmp.columns else None,
            hover_data=["user_hash", "day", "alert"],
            title="After-hours USB usage vs final_score",
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Top domains on alert days
    if not events.empty and "user_hash" in events.columns:
        ev = events.copy()
        ev["timestamp"] = _safe_datetime(ev, "timestamp")
        ev["day"] = ev["timestamp"].dt.floor("D")
        ev["domain"] = ev.get("url", pd.Series([None] * len(ev))).apply(_domain)

        alert_days = joined[joined["alert"] == True][["user_hash", "day"]].copy()
        alert_days["day"] = alert_days["day"].dt.floor("D")
        ev = ev.merge(alert_days.assign(_alert_day=True), on=["user_hash", "day"], how="inner")
        http = ev[ev["source"] == "http"].copy()
        if not http.empty:
            topd = http["domain"].value_counts().head(20).reset_index()
            topd.columns = ["domain", "visits"]
            fig = px.bar(topd, x="visits", y="domain", orientation="h", title="Top HTTP domains during alert-days")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Evaluation / F1
# ----------------------------
with eval_tab:
    st.subheader("Ground-truth evaluation")

    if evaluation is None:
        st.info(
            "No evaluation files were found. To visualize F1 in the GUI, provide a labels CSV with user or user_hash, a day/date column, and a binary label column, then run Train or Detect."
        )
    else:
        summary = evaluation["summary"]
        by_day = evaluation.get("by_day", pd.DataFrame())
        threshold_curve = evaluation.get("threshold_curve", pd.DataFrame())
        scored_eval = evaluation.get("scored", pd.DataFrame())

        st.caption(f"Labels file: {summary.get('labels_path', '')}")

        if summary.get("status") == "no_overlap":
            st.warning("A labels file was detected, but it did not match any scored user-day rows. Check the SHA-256 salt, user/user_hash values, and date granularity.")
        else:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("F1", f"{float(summary.get('f1', 0.0)):.3f}", delta=f"best {float(summary.get('best_f1', 0.0)):.3f}" if summary.get("best_f1") is not None else None)
            c2.metric("Precision", f"{float(summary.get('precision', 0.0)):.3f}")
            c3.metric("Recall", f"{float(summary.get('recall', 0.0)):.3f}")
            c4.metric("Accuracy", f"{float(summary.get('accuracy', 0.0)):.3f}")
            c5.metric("Current threshold", f"{float(summary.get('current_threshold', 0.0)):.3f}" if summary.get("current_threshold") is not None else "—")
            c6.metric("Best F1 threshold", f"{float(summary.get('best_threshold', 0.0)):.3f}" if summary.get("best_threshold") is not None else "—")

            left, right = st.columns([1.2, 0.8])
            with left:
                st.plotly_chart(
                    _eval_threshold_curve(
                        threshold_curve,
                        summary.get("current_threshold"),
                        summary.get("best_threshold"),
                    ),
                    use_container_width=True,
                )
            with right:
                st.plotly_chart(_confusion_matrix_fig(summary), use_container_width=True)

            d1, d2 = st.columns([1.1, 0.9])
            with d1:
                st.plotly_chart(_eval_daily_scores(by_day), use_container_width=True)
            with d2:
                st.markdown("#### Confusion counts")
                cm = pd.DataFrame(
                    [
                        {"bucket": "True Positive", "count": int(summary.get("tp", 0))},
                        {"bucket": "False Positive", "count": int(summary.get("fp", 0))},
                        {"bucket": "True Negative", "count": int(summary.get("tn", 0))},
                        {"bucket": "False Negative", "count": int(summary.get("fn", 0))},
                    ]
                )
                st.dataframe(cm, use_container_width=True, hide_index=True)

            if not scored_eval.empty:
                st.markdown("#### Evaluation rows")
                show_cols = [
                    "day",
                    "user_hash",
                    "final_score",
                    "threshold",
                    "severity",
                    "canary_hit",
                    "label",
                    "pred_current",
                    "top_factors",
                    "explanation",
                ]
                show_cols = [c for c in show_cols if c in scored_eval.columns]
                st.dataframe(scored_eval[show_cols].head(500), use_container_width=True, hide_index=True)

                st.download_button(
                    "Download evaluation CSV",
                    data=scored_eval[show_cols].to_csv(index=False).encode("utf-8"),
                    file_name="demf_evaluation_scored.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ----------------------------
# Model & Config
# ----------------------------
with model_tab:
    st.subheader("Model diagnostics and configuration")

    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.plotly_chart(_scatter_diag(scores), use_container_width=True)

    with c2:
        st.markdown("#### Current config")
        st.json(asdict(cfg))
        st.markdown("#### Model artifacts")
        files = ["scaler.joblib", "autoencoder.joblib", "ocsvm.joblib", "threshold.json", "feature_cols.json"]
        rows = []
        for fn in files:
            p = cfg.model_dir / fn
            rows.append({"file": fn, "exists": p.exists(), "size_kb": round(p.stat().st_size / 1024, 1) if p.exists() else None})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(f"Reports: {cfg.report_dir.resolve()}  •  Models: {cfg.model_dir.resolve()}  •  Data: {cfg.raw_dir.resolve()}")


with st.expander("How DEMF raises an alert", expanded=False):
    st.markdown(
        """
**Aggregation unit:** user-day (one record per user per calendar day)

**Signal layers (Defense-in-Depth):**
1. **Reconstruction anomaly** (autoencoder-like): unusual feature patterns vs baseline.
2. **Boundary anomaly** (One-Class SVM): points outside the normal region.
3. **Context score**: after-hours + USB + multi-PC + after-hours browsing.
4. **Canary confirmation**: if a decoy token is observed, severity becomes **CRITICAL**.

**Scores:**
- `combined_score`: normalized blend of reconstruction + SVM.
- `final_score`: `combined_score` blended with `context_score`.

**Alert rule:**
- `final_score >= threshold` OR `canary_hit == True`
"""
    )
