"""Unified Dashboard - AI-Driven Insider Threat Detection & HEADS Threat Monitoring

This dashboard combines both the Insider Threat Detection system and the HEADS 
(Hybrid Environment Authentication Anomaly Detection System) into a single 
interface with navigation between the two systems.
"""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Add both app directories to path for imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
streamlit_app_dir = project_root / "streamlit_app"
sys.path.insert(0, str(streamlit_app_dir))

# Configure page
st.set_page_config(
    page_title="Unified Threat Monitoring Dashboard",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation
if 'current_view' not in st.session_state:
    st.session_state.current_view = "AI-Driven Insider Threat Monitoring Dashboard"

# Custom CSS for navigation and styling
st.markdown("""
<style>
/* Navigation bar styling */
.nav-container {
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    padding: 1rem 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.nav-title {
    color: white;
    font-size: 1.8rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
    text-align: center;
}
.nav-subtitle {
    color: #e0e7ff;
    font-size: 0.9rem;
    text-align: center;
    margin-bottom: 1rem;
}
.stApp {
    background: #0e1117;
}
/* Button styling */
div.row-widget.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}
div.row-widget.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
}
div.row-widget.stButton > button:active {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}
/* Metrics styling */
div[data-testid="stMetric"] {
    background: #1a1f2e;
    border: 1px solid rgba(148,163,184,0.15);
    padding: 14px;
    border-radius: 14px;
}
div[data-testid="stMetricValue"] {
    color: #e6edf3;
}
div[data-testid="stMetricLabel"] {
    color: #9da7b1;
}
</style>
""", unsafe_allow_html=True)

# Navigation header
st.markdown("""
<div class="nav-container">
    <div class="nav-title">🛡️ Unified Threat Monitoring System</div>
    <div class="nav-subtitle">Integrated AI-Driven Insider Threat Detection & HEADS Anomaly Detection</div>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    st.markdown("---")
    
    # Navigation buttons
    if st.button("📊 AI-Driven Insider Threat Monitoring Dashboard", use_container_width=True):
        st.session_state.current_view = "AI-Driven Insider Threat Monitoring Dashboard"
        st.rerun()
    
    if st.button("🔴 Threat Details (HEADS)", use_container_width=True):
        st.session_state.current_view = "Threat Details"
        st.rerun()
    
    if st.button("📈 Model Evaluation (HEADS)", use_container_width=True):
        st.session_state.current_view = "Model Evaluation"
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"**Current View:**")
    st.info(st.session_state.current_view)
    
    st.markdown("---")
    st.markdown("### ℹ️ System Info")
    st.markdown("""
    **Two Integrated Systems:**
    
    1️⃣ **Insider Threat Detection**
    - GRU-based sequential anomaly detection
    - Email & login activity monitoring
    - Real-time streaming alerts
    
    2️⃣ **HEADS System**
    - Transformer + GraphSAGE + XGBoost
    - Network behavior analysis
    - Live feed visualization
    """)

# Load the appropriate dashboard based on selection
if st.session_state.current_view == "AI-Driven Insider Threat Monitoring Dashboard":
    # Load insider threat monitoring dashboard with full tabs
    st.markdown("---")
    try:
        # Import and run the insider threat dashboard components
        import time
        from datetime import datetime
        from typing import Any
        import pandas as pd
        
        from insider_gru.config import get_default_config
        from insider_gru.integrity import verify_alert_hash
        from insider_gru.streaming import load_latest_jsonl_records

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

        st.title("📊 AI-Driven Insider Threat Monitoring Dashboard")
        st.caption("Real-time insider threat detection for hybrid work environments (email + login/device) using manual + hybrid scoring.")

        # Get paths from config
        cfg = get_default_config()
        base_out = Path(cfg.output.out_dir)
        live_dir = base_out / "live"
        live_events_path = live_dir / "live_events.jsonl"
        live_alerts_path = live_dir / "live_alerts.jsonl"

        # Sidebar controls for this view
        with st.sidebar:
            st.markdown("---")
            st.markdown("### ⚙️ Dashboard Controls")
            auto_refresh = st.toggle("Auto-refresh", value=False, key="insider_auto_refresh")
            refresh_interval = st.number_input("Refresh interval (seconds)", min_value=1, max_value=300, value=5, step=1, key="insider_refresh_interval")
            max_alerts = st.number_input("Max alerts to display", min_value=10, max_value=5000, value=200, step=10, key="insider_max_alerts")

            st.markdown("### 🔍 Filters")
            event_type_filter = st.selectbox("Filter by event type", options=["All", "Email", "Login"], index=0, key="insider_event_filter")
            decision_filter = st.selectbox("Filter by decision", options=["All", "OK", "ALERT"], index=0, key="insider_decision_filter")
            min_score = st.slider("Minimum final score", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="insider_min_score")

            st.markdown("### 📂 Live Sources")
            status_df = pd.DataFrame([_file_status(live_events_path), _file_status(live_alerts_path)])
            st.dataframe(status_df[["exists", "size_kb", "modified"]], use_container_width=True, hide_index=True)

        # Load alerts
        records = _load_alert_records(live_alerts_path, limit=int(max_alerts))
        alerts_df = _alerts_to_df(records)
        filtered_df = _apply_filters(alerts_df, event_type_filter, decision_filter, float(min_score))

        # Tabs for different views
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
                st.code("""
# For email events:
python scripts/simulate_stream.py --event-type email --source Dataset/train/R2/email-train-data.csv --delay 0.25 --clear-output

# For login events:
python scripts/simulate_stream.py --event-type login --source Dataset/train/R2/device-train-data.csv --delay 0.25 --clear-output
                """)
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
                    selected_id = st.selectbox("Select an alert", options=options, format_func=_label, key="investigation_select")
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

                    selected_id = st.selectbox("Select an alert to verify", options=ordered_ids, format_func=_label, key="integrity_select")
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
                            st.success("✅ Verified: computed hash matches the ledger.")
                        elif status == "Tampered":
                            st.error("❌ Tampered: computed hash does not match the ledger.")
                        else:
                            st.warning("⚠️ Missing: no matching ledger entry found for this alert.")

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
            
            st.write("**Scoring modes:**")
            st.markdown("""
            - **Model Prediction:** final score = model probability (if available)
            - **Manual Scoring:** rule-based score from event context  
            - **Hybrid:** blend of model probability and manual score
            """)
            
            st.write("---")
            st.write(f"**Current threshold (from config):** {cfg.scoring.threshold:.2f}")
            
            st.write("**Score interpretation (high level):**")
            st.markdown("""
            - **0.00–0.39:** typically OK
            - **0.40–0.59:** borderline / review
            - **0.60–1.00:** elevated risk (likely ALERT depending on threshold)
            """)

        # Auto-refresh at the end
        if auto_refresh:
            time.sleep(float(refresh_interval))
            st.rerun()

    except Exception as e:
        st.error(f"Error loading Insider Threat Dashboard: {e}")
        st.exception(e)

elif st.session_state.current_view == "Threat Details":
    # Load HEADS Threat Details dashboard
    st.markdown("---")
    try:
        import time
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime
        from threat_mapping import categorize_threats, summary_counts, PROFILES
        from data_sources import list_csv_instances

        # Initialize dark mode in session state
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False

        # Sidebar controls for HEADS data
        with st.sidebar:
            st.markdown("---")
            st.header("⚙️ Data Configuration")

            feed_mode = st.selectbox(
                "Select data source",
                ["Live feed", "Test instance (dropdown)", "Default scored_events.csv"],
                index=0,
                key="threat_details_feed"
            )

            profile = st.selectbox("Threat mapping profile", list(PROFILES.keys()), index=1, key="threat_details_profile")
            min_prob = st.slider("Min probability (if available)", 0.0, 1.0, 0.50, 0.01, key="threat_details_minprob")
            rows_to_show = st.slider("Rows to show", 50, 1000, 300, 50, key="threat_details_rows")

            st.divider()
            dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode, key="threat_details_dark")
            st.session_state.dark_mode = dark_mode
            auto_refresh = st.toggle("Auto refresh", value=False, key="threat_details_autorefresh")
            refresh_sec = st.slider("Refresh interval (sec)", 2, 30, 5, key="threat_details_refreshsec")
            
            if st.button("🔄 Refresh Now", use_container_width=True, key="threat_details_refresh"):
                st.rerun()

        # Custom CSS for dark mode
        st.markdown(f"""
        <style>
        .block-container {{ padding-top: 1.5rem; padding-bottom: 1.2rem; }}
        div[data-testid="stMetric"] {{
          background: {'#0F172A' if st.session_state.dark_mode else '#FFFFFF'};
          border: 1px solid {'rgba(148,163,184,0.15)' if st.session_state.dark_mode else '#E5E7EB'};
          padding: 14px;
          border-radius: 14px;
        }}
        .card {{
          background: {'#0F172A' if st.session_state.dark_mode else '#FFFFFF'};
          border: 1px solid {'rgba(148,163,184,0.15)' if st.session_state.dark_mode else '#E5E7EB'};
          padding: 14px;
          border-radius: 14px;
        }}
        .small {{ color: {'#94A3B8' if st.session_state.dark_mode else '#6B7280'}; font-size: 0.9rem; }}
        hr {{ margin: 0.6rem 0; border-color: {'rgba(148,163,184,0.15)' if st.session_state.dark_mode else '#E5E7EB'}; }}
        </style>
        """, unsafe_allow_html=True)

        # Load data
        df = None
        source_label = ""

        if feed_mode == "Live feed":
            p = project_root / "data" / "live" / "live_events.csv"
            source_label = "LIVE"
            if p.exists():
                try:
                    df = pd.read_csv(p)
                except Exception:
                    df = None
            if df is None or len(df) == 0:
                p2 = project_root / "data" / "processed" / "scored_events.csv"
                source_label = "LIVE (empty) → SCORED"
                if not p2.exists():
                    st.error("No live data and scored_events.csv not found. Run training notebook first.")
                    st.stop()
                df = pd.read_csv(p2)
        elif feed_mode == "Default scored_events.csv":
            p = project_root / "data" / "processed" / "scored_events.csv"
            source_label = "SCORED (default)"
            if not p.exists():
                st.error("Default scored_events.csv not found. Run training notebook first.")
                st.stop()
            df = pd.read_csv(p)
        else:  # Test instance dropdown
            files = list_csv_instances()
            if not files:
                st.error("No test instances found in data/processed/ or data/test/.")
                st.stop()
            labels = []
            idx_default = 0
            for i, p_file in enumerate(files):
                tag = " (default)" if p_file.name == "scored_events.csv" else ""
                labels.append(p_file.name + tag)
                if p_file.name == "scored_events.csv":
                    idx_default = i
            chosen = st.selectbox("Choose test instance", labels, index=idx_default, key="threat_details_instance")
            chosen_path = files[labels.index(chosen)]
            source_label = f"TEST: {chosen_path.name}"
            df = pd.read_csv(chosen_path)

        # Apply threat categorization
        df = categorize_threats(df, profile=profile)

        # Probability filter
        if "xgb_proba" in df.columns:
            df["xgb_proba"] = pd.to_numeric(df["xgb_proba"], errors="coerce").fillna(0.0)
            if min_prob > 0:
                df = df[df["xgb_proba"] >= min_prob]

        # Build alerts view
        alerts_df = df[df["threat_category"] != "Normal"].copy()

        # Header with title and source
        st.markdown(
            f"""
            <div style='padding-top:0.5rem; padding-bottom:0.5rem; width:100%;'>
                <span style='font-size:2.2rem; font-weight:700; vertical-align:middle;'>🔴 Threat Details</span>
                <span class='small' style='font-size:1.1rem; margin-left:0.5rem; vertical-align:middle;'>({source_label})</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Top KPIs
        total_users = df["src_ip"].nunique() if "src_ip" in df.columns else 0
        alerts = len(alerts_df)
        avg_score = float(df["xgb_proba"].mean()) if "xgb_proba" in df.columns and len(df) > 0 else float(df.get("temporal_score", pd.Series([0.0])).mean())

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Users Analysed", f"{total_users:,}")
        k2.metric("High Risk", f"{int((alerts_df['threat_category'] != 'Normal').sum()):,}" if len(alerts_df) else "0")
        k3.metric("Critical", f"{int((alerts_df['xgb_proba'] >= 0.9).sum()):,}" if "xgb_proba" in alerts_df.columns else "0")
        k4.metric("Medium Risk", f"{int(((alerts_df.get('xgb_proba', 0) >= 0.6) & (alerts_df.get('xgb_proba', 0) < 0.9)).sum()):,}" if "xgb_proba" in alerts_df.columns else "0")
        k5.metric("Average Score", f"{avg_score:.3f}")

        # Risk level breakdown
        risk_levels = ["Very Low", "Low", "Medium", "High", "Critical"]
        if "xgb_proba" in df.columns:
            p_vals = df["xgb_proba"].fillna(0.0)
            risk = pd.cut(p_vals, bins=[-1, 0.2, 0.4, 0.6, 0.8, 1.0], labels=risk_levels)
        else:
            risk = pd.Series(["Low"] * len(df))

        r_counts = risk.value_counts().reindex(risk_levels, fill_value=0)
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("VERY LOW", int(r_counts["Very Low"]))
        r2.metric("LOW", int(r_counts["Low"]))
        r3.metric("MEDIUM", int(r_counts["Medium"]))
        r4.metric("HIGH", int(r_counts["High"]))
        r5.metric("CRITICAL", int(r_counts["Critical"]))

        # Tabs
        tab_overview, tab_live, tab_users, tab_table = st.tabs(["Overview", "Threat Preview", "Category Views", "Analyst Table"])

        with tab_overview:
            c1, c2, c3 = st.columns([1.1, 1.1, 1.8])

            with c1:
                summ = summary_counts(df, "threat_category")
                summ = summ[summ["threat_category"] != "Normal"]
                if len(summ) == 0:
                    st.markdown("<div class='card'><b>Threat Categories</b><br/><span class='small'>No alerts</span></div>", unsafe_allow_html=True)
                else:
                    st.plotly_chart(px.pie(summ, names="threat_category", values="count", hole=0.65, title="Threat Categories"), use_container_width=True)

            with c2:
                summ2 = summary_counts(df, "threat_category")
                st.plotly_chart(px.bar(summ2, x="threat_category", y="count", title="Threat Category Count"), use_container_width=True)

            with c3:
                if "date" in df.columns:
                    trend = df.groupby("date").size().reset_index(name="count")
                    st.plotly_chart(px.area(trend, x="date", y="count", title="Logs Trend"), use_container_width=True)
                else:
                    st.markdown("<div class='card'><b>Logs Trend</b><br/><span class='small'>No timestamp available</span></div>", unsafe_allow_html=True)

            st.divider()
            g1, g2, g3 = st.columns(3)
            cats = ["Suspicious Login", "Privilege Abuse", "Account Takeover"]

            def cat_trend(cat):
                d = df[df["threat_category"] == cat]
                if "date" not in d.columns or len(d) == 0:
                    return None
                return d.groupby("date").size().reset_index(name="count")

            for col, cat in zip([g1, g2, g3], cats):
                with col:
                    t = cat_trend(cat)
                    if t is None:
                        st.markdown(f"<div class='card'><b>{cat}</b><br/><span class='small'>No data</span></div>", unsafe_allow_html=True)
                    else:
                        st.plotly_chart(px.line(t, x="date", y="count", title=cat), use_container_width=True)

        with tab_live:
            st.markdown("<span class='small'>Threat Preview shows the newest alerts first (based on timestamp).</span>", unsafe_allow_html=True)
            view = alerts_df.copy()
            if "timestamp" in view.columns:
                view["timestamp"] = pd.to_datetime(view["timestamp"], errors="coerce")
                view = view.sort_values("timestamp", ascending=False)
            cols_show = [c for c in ["timestamp", "src_ip", "dst_ip", "protocol", "src_port", "dst_port", "xgb_proba", "threat_category", "temporal_score", "relational_score"] if c in view.columns]
            st.dataframe(view[cols_show].head(rows_to_show), use_container_width=True)

        with tab_users:
            cat = st.selectbox("Select category", ["Suspicious Login", "Privilege Abuse", "Account Takeover"], key="threat_details_cat")
            d = df[df["threat_category"] == cat].copy()
            st.markdown(f"#### {cat} Overview", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Events", f"{len(d):,}")
            c2.metric("Actors", d["src_ip"].nunique() if "src_ip" in d.columns else 0)
            c3.metric("Targets", d["dst_ip"].nunique() if "dst_ip" in d.columns else 0)

            colA, colB = st.columns(2)
            if "src_ip" in d.columns and len(d) > 0:
                topa = d["src_ip"].astype(str).value_counts().head(10).reset_index()
                topa.columns = ["src_ip", "count"]
                colA.plotly_chart(px.bar(topa, x="src_ip", y="count", title="Top Actors"), use_container_width=True)
            if "dst_ip" in d.columns and len(d) > 0:
                topt = d["dst_ip"].astype(str).value_counts().head(10).reset_index()
                topt.columns = ["dst_ip", "count"]
                colB.plotly_chart(px.bar(topt, x="dst_ip", y="count", title="Top Targets"), use_container_width=True)

            if "date" in d.columns and len(d) > 0:
                tr = d.groupby("date").size().reset_index(name="count")
                st.plotly_chart(px.line(tr, x="date", y="count", title=f"{cat} trend"), use_container_width=True)

        with tab_table:
            st.markdown("<span class='small'>Analyst view: filter by risk level and minimum score.</span>", unsafe_allow_html=True)
            min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01, key="threat_details_minscore")
            risk_sel = st.selectbox("Risk level", ["All"] + risk_levels, key="threat_details_risk")

            tdf = df.copy()
            if "xgb_proba" in tdf.columns:
                tdf = tdf[tdf["xgb_proba"].fillna(0.0) >= min_score]
            if risk_sel != "All":
                tdf = tdf[risk == risk_sel]

            cols_show = [c for c in ["timestamp", "src_ip", "dst_ip", "protocol", "src_port", "dst_port", "xgb_proba", "threat_category"] if c in tdf.columns]
            if "timestamp" in tdf.columns:
                tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
                tdf = tdf.sort_values("timestamp", ascending=False)
            st.dataframe(tdf[cols_show].head(rows_to_show), use_container_width=True)

        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_sec)
            st.rerun()

    except Exception as e:
        st.error(f"Error loading HEADS Threat Details: {e}")
        st.exception(e)

elif st.session_state.current_view == "Model Evaluation":
    # Load HEADS Model Evaluation dashboard
    st.markdown("---")
    try:
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        from data_sources import list_csv_instances

        st.title("📈 Model Evaluation (HEADS System)")

        # Sidebar controls
        with st.sidebar:
            st.markdown("---")
            st.header("⚙️ Evaluation Configuration")

        st.markdown("<span class='small'>Evaluate scored test instances (data/test/) or the default scored_events.csv.</span>", unsafe_allow_html=True)
        
        files = list_csv_instances()
        if not files:
            st.warning("No CSV instances found in data/test/ or data/processed/.")
        else:
            labels = []
            idx = 0
            for i, p in enumerate(files):
                tag = " (default)" if p.name == "scored_events.csv" else ""
                labels.append(p.parent.name + "/" + p.name + tag)
                if p.name == "scored_events.csv":
                    idx = i
            
            chosen = st.selectbox("Evaluation dataset (scored CSV)", labels, index=idx, key="eval_select")
            threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="eval_thresh")
            
            # File selection option for comparison
            st.markdown("---")
            st.markdown("<span class='small'>Pick files like test_v1_random__xgb_v2.csv (data/test/) to compare model variants.</span>", unsafe_allow_html=True)
            
            p = files[labels.index(chosen)]
            dfe = pd.read_csv(p)
            
            if not {"label", "xgb_proba"}.issubset(dfe.columns):
                st.error("Selected file must include label and xgb_proba.")
            else:
                y_true = pd.to_numeric(dfe["label"], errors="coerce").fillna(0).astype(int).to_numpy()
                y_score = pd.to_numeric(dfe["xgb_proba"], errors="coerce").fillna(0).astype(float).to_numpy()
                y_pred = (y_score >= threshold).astype(int)

                # Metrics
                k1, k2, k3, k4 = st.columns(4)
                if len(np.unique(y_true)) > 1:
                    try:
                        from sklearn.metrics import roc_auc_score, average_precision_score
                        k1.metric("ROC-AUC", f"{roc_auc_score(y_true, y_score):.3f}")
                        k2.metric("PR-AUC", f"{average_precision_score(y_true, y_score):.3f}")
                    except ImportError:
                        k1.metric("ROC-AUC", "Scikit-learn Error")
                        k2.metric("PR-AUC", "Scikit-learn Error")
                else:
                    k1.metric("ROC-AUC", "N/A")
                    k2.metric("PR-AUC", "N/A")
                k3.metric("Attack rate", f"{(y_true == 1).mean() * 100:.2f}%")
                k4.metric("Predicted positive", f"{(y_pred == 1).mean() * 100:.2f}%")

                # Probability distribution
                st.plotly_chart(px.histogram(dfe, x="xgb_proba", nbins=60, title="Probability distribution"), use_container_width=True)

                # Label share pie chart
                label_counts = pd.DataFrame({
                    'label': ['0', '1'],
                    'count': [(y_true == 0).sum(), (y_true == 1).sum()]
                })
                st.plotly_chart(px.pie(label_counts, names='label', values='count', title='Label share', hole=0.4), use_container_width=True)

                if len(np.unique(y_true)) > 1:
                    try:
                        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, classification_report
                        
                        # Confusion matrix
                        cm = confusion_matrix(y_true, y_pred)
                        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
                        fig = go.Figure(data=go.Heatmap(
                            z=cm_norm,
                            x=["Pred 0", "Pred 1"],
                            y=["True 0", "True 1"],
                            colorscale='Blues',
                            text=cm,
                            texttemplate='%{text}',
                            textfont={"size": 16}
                        ))
                        fig.update_layout(title="Confusion Matrix (row-normalized)", height=400)
                        st.plotly_chart(fig, use_container_width=True)

                        # ROC Curve
                        fpr, tpr, _ = roc_curve(y_true, y_score)
                        st.plotly_chart(px.line(
                            pd.DataFrame({"fpr": fpr, "tpr": tpr}),
                            x="fpr",
                            y="tpr",
                            title="ROC Curve"
                        ), use_container_width=True)

                        # Precision-Recall Curve
                        prec, rec, _ = precision_recall_curve(y_true, y_score)
                        st.plotly_chart(px.line(
                            pd.DataFrame({"recall": rec, "precision": prec}),
                            x="recall",
                            y="precision",
                            title="Precision-Recall Curve"
                        ), use_container_width=True)

                        # Classification Report
                        st.markdown("### Classification Report")
                        report = classification_report(y_true, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)

                    except ImportError:
                        st.error("Scikit-learn is not available. Please check the installation.")
                else:
                    st.info("Need both classes in evaluation set to draw ROC/PR curves.")

    except Exception as e:
        st.error(f"Error loading HEADS Model Evaluation: {e}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9da7b1; padding: 1rem;'>
    <p>Unified Threat Monitoring System | AI-Driven Security Analytics</p>
    <p style='font-size: 0.8rem;'>Powered by GRU + Transformer + GraphSAGE + XGBoost</p>
</div>
""", unsafe_allow_html=True)
