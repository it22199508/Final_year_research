from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from insider_gru.config import get_default_config
from insider_gru.integrity import verify_alert_hash
from insider_gru.streaming import load_latest_jsonl_records


st.set_page_config(page_title="AI-Driven Insider Threat Monitoring Dashboard", layout="wide")


def _apply_dark_style() -> None:
    """Lightweight dark styling without over-customizing Streamlit."""

    st.markdown(
        """
<style>
/* Keep it subtle: better contrast in dark mode */
.stApp { background: #0e1117; }
.stMarkdown, .stText, .stCaption, .stDataFrame { color: #e6edf3; }
div[data-testid="stMetricValue"] { color: #e6edf3; }
div[data-testid="stMetricLabel"] { color: #9da7b1; }
</style>
""",
        unsafe_allow_html=True,
    )


def _paths_from_config() -> tuple[Path, Path, Path]:
    cfg = get_default_config()
    base_out = Path(cfg.output.out_dir)
    live_dir = base_out / "live"
    return live_dir, live_dir / "live_events.jsonl", live_dir / "live_alerts.jsonl"


def _file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "size_kb": 0.0, "modified": "-"}
    try:
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).isoformat(sep=" ", timespec="seconds")
        return {
            "path": str(path),
            "exists": True,
            "size_kb": float(stat.st_size) / 1024.0,
            "modified": modified,
        }
    except Exception:
        return {"path": str(path), "exists": True, "size_kb": None, "modified": "-"}


def _load_alert_records(alerts_path: Path, limit: int) -> list[dict]:
    return load_latest_jsonl_records(alerts_path, limit=int(limit))


def _alerts_to_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=["processed_at", "event_type", "final_score", "confidence_label", "decision", "alert_id"]
        )
    df = pd.DataFrame(records)
    for c in ["processed_at", "event_type", "final_score", "confidence_label", "decision", "alert_id"]:
        if c not in df.columns:
            df[c] = None
    df["processed_at"] = pd.to_datetime(df["processed_at"], errors="coerce")
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    df["event_type"] = df["event_type"].astype(str).str.lower()
    df["decision"] = df["decision"].astype(str).str.upper()
    df = df.sort_values("processed_at", ascending=False, na_position="last")
    return df


def _apply_filters(
    df: pd.DataFrame,
    event_type_filter: str,
    decision_filter: str,
    min_score: float,
) -> pd.DataFrame:
    out = df.copy()
    if event_type_filter != "All":
        out = out[out["event_type"] == event_type_filter.lower()]
    if decision_filter != "All":
        out = out[out["decision"] == decision_filter.upper()]
    out = out[out["final_score"].fillna(0.0) >= float(min_score)]
    return out


def _maybe_autorefresh(enabled: bool, interval_seconds: float) -> None:
    """Streamlit-friendly auto-refresh.

    Streamlit doesn't include a native timer callback; a simple pattern is to
    sleep at the end of the run and then call `st.rerun()`.
    """

    if not enabled:
        return

    s = float(interval_seconds)
    if s <= 0:
        return

    # Keep the pause at the end of the run so UI renders first.
    time.sleep(s)
    st.rerun()


def main() -> None:
    _apply_dark_style()

    st.title("AI-Driven Insider Threat Monitoring Dashboard")
    st.caption("Real-time insider threat detection for hybrid work environments (email + login/device) using manual + hybrid scoring.")

    live_dir, live_events_path, live_alerts_path = _paths_from_config()

    # Sidebar controls
    st.sidebar.header("Live Controls")
    auto_refresh = st.sidebar.toggle("Auto-refresh", value=False)
    refresh_interval = st.sidebar.number_input("Refresh interval (seconds)", min_value=1, max_value=300, value=5, step=1)
    max_alerts = st.sidebar.number_input("Max alerts to display", min_value=10, max_value=5000, value=200, step=10)

    st.sidebar.header("Filters")
    event_type_filter = st.sidebar.selectbox("Filter by event type", options=["All", "Email", "Login"], index=0)
    decision_filter = st.sidebar.selectbox("Filter by decision", options=["All", "OK", "ALERT"], index=0)
    min_score = st.sidebar.slider("Minimum final score", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    st.sidebar.header("Live Sources")
    st.sidebar.write("Expected JSONL outputs:")
    st.sidebar.code(str(live_events_path))
    st.sidebar.code(str(live_alerts_path))

    status_df = pd.DataFrame([_file_status(live_events_path), _file_status(live_alerts_path)])
    st.sidebar.dataframe(status_df[["exists", "size_kb", "modified"]], use_container_width=True, hide_index=True)

    # Load alerts
    records = _load_alert_records(live_alerts_path, limit=int(max_alerts))
    alerts_df = _alerts_to_df(records)
    filtered_df = _apply_filters(alerts_df, event_type_filter, decision_filter, float(min_score))

    tabs = st.tabs(["Overview", "Live Alerts", "Alert Investigation", "Integrity Verification", "Model / Scoring Summary"])

    with tabs[0]:
        st.subheader("Overview")

        total_loaded = int(len(alerts_df))
        alert_count = int((alerts_df["decision"] == "ALERT").sum()) if total_loaded else 0
        avg_score = float(alerts_df["final_score"].mean()) if total_loaded else 0.0
        email_count = int((alerts_df["event_type"] == "email").sum()) if total_loaded else 0
        login_count = int((alerts_df["event_type"] == "login").sum()) if total_loaded else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total alerts loaded", total_loaded)
        c2.metric("ALERT decisions", alert_count)
        c3.metric("Average final score", f"{avg_score:.3f}")
        c4.metric("Email alerts", email_count)
        c5.metric("Login alerts", login_count)

        if total_loaded == 0:
            st.info("No alerts loaded yet. Start the simulator to produce outputs/live/live_alerts.jsonl.")
        else:
            st.write("Recent alerts")
            recent_cols = ["processed_at", "event_type", "final_score", "confidence_label", "decision", "alert_id"]
            st.dataframe(filtered_df[recent_cols].head(25), use_container_width=True, hide_index=True)

            # Simple score distribution chart
            scores = filtered_df["final_score"].dropna().clip(0.0, 1.0)
            if len(scores) > 0:
                bins = pd.cut(scores, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
                hist = bins.value_counts().sort_index()
                st.write("Score distribution")
                # Streamlit/Altair expects index values to be JSON-serializable.
                # Pandas names the reset_index() column based on the Series/index name.
                hist_df = hist.rename("count").reset_index()
                bin_col = str(hist_df.columns[0])
                hist_df["bin"] = hist_df[bin_col].astype(str)
                hist_df = hist_df[["bin", "count"]].set_index("bin")
                st.bar_chart(hist_df)

    with tabs[1]:
        st.subheader("Live Alerts")

        if len(alerts_df) == 0:
            st.info("No alerts available.")
        else:
            view = filtered_df.copy()
            view_cols = ["processed_at", "event_type", "final_score", "confidence_label", "decision", "alert_id"]
            st.dataframe(view[view_cols], use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Alert Investigation")

        if not records:
            st.info("No alert records loaded.")
        else:
            # Build selection labels from the filtered set, but keep mapping to full record.
            id_to_record = {str(r.get("alert_id")): r for r in records if isinstance(r, dict) and r.get("alert_id")}
            if not id_to_record:
                st.warning("Alerts are missing alert_id fields.")
            else:
                # Use filtered order if available.
                ordered_ids = [str(x) for x in filtered_df["alert_id"].dropna().tolist() if str(x) in id_to_record]
                if not ordered_ids:
                    ordered_ids = list(id_to_record.keys())

                def _label(aid: str) -> str:
                    r = id_to_record.get(aid, {})
                    ts = str(r.get("processed_at") or "-")
                    et = str(r.get("event_type") or "-")
                    decision = str(r.get("decision") or "-")
                    return f"{ts} | {et} | {decision} | {aid[:8]}"

                options = ordered_ids
                selected_id = st.selectbox("Select an alert", options=options, format_func=_label)
                selected = id_to_record.get(str(selected_id), {})

                # Core fields
                st.write("Summary")
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Decision", str(selected.get("decision", "-")))
                s2.metric("Final score", f"{float(selected.get('final_score', 0.0)):.3f}" if selected.get("final_score") is not None else "-")
                s3.metric("Confidence", str(selected.get("confidence_label", "-")))
                s4.metric("Manual score", f"{float(selected.get('manual_score', 0.0)):.3f}" if selected.get("manual_score") is not None else "-")
                ms = selected.get("model_score")
                s5.metric("Model score", "-" if ms is None else f"{float(ms):.3f}")

                st.write("Cleaned inputs")
                st.json(selected.get("cleaned_inputs", {}))

                st.write("Engineered features")
                st.json(selected.get("engineered_features", {}))

                st.write("Triggered rules")
                st.json(selected.get("triggered_rules", []))

    with tabs[3]:
        st.subheader("Integrity Verification")

        cfg = get_default_config()
        ledger_path = Path(cfg.integrity.ledger_path)
        hash_algorithm = str(cfg.integrity.hash_algorithm or "sha256")
        st.caption(f"Ledger: {ledger_path} ({hash_algorithm})")

        ledger_status = _file_status(ledger_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("Ledger exists", "Yes" if ledger_status.get("exists") else "No")
        c2.metric("Ledger size (KB)", f"{float(ledger_status.get('size_kb') or 0.0):.1f}" if ledger_status.get("exists") else "0.0")
        c3.metric("Ledger modified", str(ledger_status.get("modified") or "-"))

        if not records:
            st.info("No alert records loaded yet. Start the simulator to produce outputs/live/live_alerts.jsonl.")
        else:
            id_to_record = {str(r.get("alert_id")): r for r in records if isinstance(r, dict) and r.get("alert_id")}
            if not id_to_record:
                st.warning("Alerts are missing alert_id fields; cannot verify integrity.")
            else:
                ordered_ids = [str(x) for x in alerts_df["alert_id"].dropna().tolist() if str(x) in id_to_record]
                if not ordered_ids:
                    ordered_ids = list(id_to_record.keys())

                def _label(aid: str) -> str:
                    r = id_to_record.get(aid, {})
                    ts = str(r.get("processed_at") or "-")
                    et = str(r.get("event_type") or "-")
                    decision = str(r.get("decision") or "-")
                    return f"{ts} | {et} | {decision} | {aid[:8]}"

                selected_id = st.selectbox("Select an alert to verify", options=ordered_ids, format_func=_label)
                selected = id_to_record.get(str(selected_id), {})

                # Display key fields
                st.write("Selected alert")
                summary = {
                    "alert_id": selected.get("alert_id"),
                    "event_type": selected.get("event_type"),
                    "final_score": selected.get("final_score"),
                    "decision": selected.get("decision"),
                    "processed_at": selected.get("processed_at"),
                    "integrity_hash": selected.get("integrity_hash"),
                }
                st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)

                # Verify against ledger
                try:
                    result = verify_alert_hash(
                        selected,
                        ledger_path=ledger_path,
                        hash_algorithm=hash_algorithm,
                        # Simulator excludes integrity_hash before hashing, then injects it.
                        exclude_fields=["integrity_hash"],
                    )
                except Exception as e:
                    st.error(f"Integrity verification failed: {type(e).__name__}: {e}")
                else:
                    status = str(result.get("status") or "Missing")
                    stored_hash = result.get("stored_hash")
                    computed_hash = result.get("computed_hash")
                    matched = bool(result.get("matched"))

                    if status == "Verified":
                        st.success("Verified: computed hash matches the ledger.")
                    elif status == "Tampered":
                        st.error("Tampered: computed hash does not match the ledger.")
                    else:
                        st.warning("Missing: no matching ledger entry found for this alert.")

                    st.write("Details")
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Status", status)
                    d2.metric("Hashes match", "Yes" if matched else "No")
                    d3.metric("Hash algorithm", hash_algorithm)

                    st.write("Stored hash (ledger)")
                    st.code("-" if stored_hash in (None, "") else str(stored_hash))
                    st.write("Computed hash (from selected alert)")
                    st.code("-" if computed_hash in (None, "") else str(computed_hash))

                with st.expander("Raw selected alert JSON", expanded=False):
                    st.json(selected)

    with tabs[4]:
        st.subheader("Model / Scoring Summary")
        cfg = get_default_config()
        st.write(
            "Scoring modes:\n"
            "- Model Prediction: final score = model probability (if available)\n"
            "- Manual Scoring: rule-based score from event context\n"
            "- Hybrid: blend of model probability and manual score"
        )
        st.write("---")
        st.write(f"Current threshold (from config): {cfg.scoring.threshold:.2f}")
        st.write("Score interpretation (high level):")
        st.write("- 0.00–0.39: typically OK")
        st.write("- 0.40–0.59: borderline / review")
        st.write("- 0.60–1.00: elevated risk (likely ALERT depending on threshold)")

    _maybe_autorefresh(bool(auto_refresh), float(refresh_interval))


if __name__ == "__main__":
    main()
