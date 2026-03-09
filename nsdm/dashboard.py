# src/dashboard.py

import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    REALTIME_ALERTS_FILE,
    TEST_DASHBOARD_FILE,
    THREAT_THRESHOLD,
    CRITICAL_THRESHOLD,
)

st.set_page_config(
    page_title="NSADM Security Dashboard",
    layout="wide",
    page_icon="🚨"
)

top_left, top_right = st.columns([5, 1])

with top_left:
    st.title("🚨 NSADM Insider Threat Monitoring Dashboard")
    st.caption("CERT r4.2 | Unseen Test Users | Analyst-Friendly View")

with top_right:
    if st.button("🔄 Refresh Live Alerts", use_container_width=True):
        st.rerun()


def load_live_alerts():
    if os.path.exists(REALTIME_ALERTS_FILE):
        try:
            with open(REALTIME_ALERTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    if "threat_score" in df.columns:
                        df["threat_score"] = pd.to_numeric(df["threat_score"], errors="coerce").fillna(0)
                    if "alert_id" in df.columns:
                        df["alert_id"] = pd.to_numeric(df["alert_id"], errors="coerce").fillna(0).astype(int)

                    sort_cols = [c for c in ["alert_id", "timestamp"] if c in df.columns]
                    if sort_cols:
                        df = df.sort_values(sort_cols, ascending=False)

                    return df
        except Exception as e:
            st.error(f"Failed to load live alerts: {e}")
    return pd.DataFrame()


def load_test_dashboard():
    if os.path.exists(TEST_DASHBOARD_FILE):
        try:
            df = pd.read_csv(TEST_DASHBOARD_FILE)
            if "threat_score" in df.columns:
                df["threat_score"] = pd.to_numeric(df["threat_score"], errors="coerce").fillna(0)
            if "rf_threat_probability" in df.columns:
                df["rf_threat_probability"] = pd.to_numeric(df["rf_threat_probability"], errors="coerce").fillna(0)
            if "lof_anomaly_score" in df.columns:
                df["lof_anomaly_score"] = pd.to_numeric(df["lof_anomaly_score"], errors="coerce").fillna(0)
            return df.sort_values("threat_score", ascending=False)
        except Exception as e:
            st.error(f"Failed to load test dashboard data: {e}")
    return pd.DataFrame()


live_df = load_live_alerts()
full_df = load_test_dashboard()

if full_df.empty:
    st.error("Test dashboard data not found. Run predict_test_users.py first.")
    st.stop()

high_count = int((full_df["threat_score"] >= THREAT_THRESHOLD).sum())
critical_count = int((full_df["threat_score"] >= CRITICAL_THRESHOLD).sum())
medium_count = int(((full_df["threat_score"] >= 0.5) & (full_df["threat_score"] < THREAT_THRESHOLD)).sum())
low_count = int(((full_df["threat_score"] >= 0.3) & (full_df["threat_score"] < 0.5)).sum())
very_low_count = int((full_df["threat_score"] < 0.3).sum())
avg_score = float(full_df["threat_score"].mean()) if len(full_df) else 0.0

last_alert_time = "N/A"
latest_alert_id = "N/A"

if not live_df.empty:
    if "timestamp" in live_df.columns and live_df["timestamp"].notna().any():
        last_alert_time = live_df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")
    if "alert_id" in live_df.columns:
        latest_alert_id = str(int(live_df["alert_id"].max()))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Users Analysed", len(full_df))
c2.metric("High Risk", high_count)
c3.metric("Critical", critical_count)
c4.metric("Latest Alert ID", latest_alert_id)
c5.metric("Last Alert Time", last_alert_time)

st.divider()

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("VERY LOW", very_low_count)
r2.metric("LOW", low_count)
r3.metric("MEDIUM", medium_count)
r4.metric("HIGH", high_count)
r5.metric("CRITICAL", critical_count)

st.divider()

f1, f2, f3 = st.columns(3)

risk_filter = f1.selectbox(
    "Filter by Risk Level",
    ["ALL", "VERY LOW", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
    index=0
)

min_score = f2.slider("Minimum Threat Score", 0.0, 1.0, 0.0, 0.05)
max_rows = f3.slider("Rows to Show", 5, 50, 15, 5)

filtered = full_df.copy()
filtered = filtered[filtered["threat_score"] >= min_score]

if risk_filter != "ALL" and "risk_level" in filtered.columns:
    filtered = filtered[filtered["risk_level"].astype(str) == risk_filter]

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Live Alerts",
    "All Test Users",
    "Analyst Table"
])

with tab1:
    left, right = st.columns(2)

    with left:
        fig1 = px.histogram(
            full_df,
            x="threat_score",
            nbins=20,
            title="Threat Score Distribution (All Test Users)"
        )
        fig1.add_vline(x=0.5, line_dash="dash", line_color="gold")
        fig1.add_vline(x=THREAT_THRESHOLD, line_dash="dash", line_color="orange")
        fig1.add_vline(x=CRITICAL_THRESHOLD, line_dash="dash", line_color="red")
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        if "risk_level" in full_df.columns:
            risk_counts = full_df["risk_level"].astype(str).value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            fig2 = px.pie(
                risk_counts,
                values="Count",
                names="Risk Level",
                title="Risk Level Breakdown"
            )
            st.plotly_chart(fig2, use_container_width=True)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("After-hours Users", int((full_df["after_hours_logons"] > 0).sum()) if "after_hours_logons" in full_df.columns else 0)
    b2.metric("Foreign-PC Users", int((full_df["foreign_pc_logons"] > 0).sum()) if "foreign_pc_logons" in full_df.columns else 0)
    b3.metric("USB-related Users", int((full_df["usb_connects"] > 0).sum()) if "usb_connects" in full_df.columns else 0)
    b4.metric("File-copy Users", int((full_df["copy_to_removable"] > 0).sum()) if "copy_to_removable" in full_df.columns else 0)

with tab2:
    st.subheader("🚨 Live Security Alerts")
    st.caption("Click Refresh Live Alerts to load new alerts from the monitor")

    if live_df.empty:
        st.info("No alerts detected yet. Start realtime_monitor.py")
    else:
        latest_alerts = live_df.head(max_rows)

        for _, row in latest_alerts.iterrows():
            score = float(row["threat_score"])
            risk = str(row.get("risk_level", "UNKNOWN"))
            user = str(row.get("user_id", "UNKNOWN"))
            assigned_pc = str(row.get("assigned_pc", "UNKNOWN"))
            last_seen_pc = str(row.get("last_seen_pc", "UNKNOWN"))
            alert_id = row.get("alert_id", "N/A")

            ts = row.get("timestamp")
            ts_text = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(ts) else "N/A"

            if risk == "CRITICAL":
                icon = "🔴"
            elif risk == "HIGH":
                icon = "🟠"
            elif risk == "MEDIUM":
                icon = "🟡"
            else:
                icon = "🟢"

            with st.container(border=True):
                a, b = st.columns([4, 1])

                with a:
                    st.markdown(f"### {icon} Alert #{alert_id} — {risk}")
                    st.write(f"**User ID:** {user}")
                    st.write(f"**Assigned PC:** {assigned_pc}")
                    st.write(f"**Last Seen PC:** {last_seen_pc}")
                    st.write(f"**Time:** {ts_text}")

                    d1, d2, d3 = st.columns(3)
                    d1.metric("Threat Score", f"{score:.3f}")
                    d2.metric("RF Score", f"{float(row.get('rf_threat_probability', 0)):.3f}")
                    d3.metric("LOF Score", f"{float(row.get('lof_anomaly_score', 0)):.3f}")

                    indicators = []
                    if row.get("after_hours_logons", 0) > 0:
                        indicators.append("🌙 After-hours logins")
                    if row.get("foreign_pc_logons", 0) > 0:
                        indicators.append("🖥 Foreign PC access")
                    if row.get("usb_connects", 0) > 0:
                        indicators.append("🔌 USB device usage")
                    if row.get("external_emails", 0) > 0:
                        indicators.append("📧 External email activity")
                    if row.get("copy_to_removable", 0) > 0:
                        indicators.append("💾 File copied to removable media")
                    if row.get("http_uploads", 0) > 0:
                        indicators.append("⬆ Suspicious upload activity")

                    if indicators:
                        st.write("**Indicators:**")
                        for ind in indicators:
                            st.write(f"- {ind}")
                    else:
                        st.write("**Indicators:** No strong indicators")

                with b:
                    st.metric("Severity", risk)
                    st.progress(min(max(score, 0.0), 1.0), text=f"{score:.1%}")

with tab3:
    st.subheader("All Test Users (Includes Low / Medium / High)")

    summary_cols = [
        "user_id",
        "assigned_pc",
        "last_seen_pc",
        "threat_score",
        "risk_level",
        "predicted_threat",
        "after_hours_logons",
        "foreign_pc_logons",
        "usb_connects",
        "external_emails",
        "http_uploads",
        "copy_to_removable",
    ]
    summary_cols = [c for c in summary_cols if c in filtered.columns]

    st.dataframe(
        filtered[summary_cols].head(max_rows),
        use_container_width=True,
        hide_index=True,
        height=450
    )

with tab4:
    st.subheader("Analyst Investigation Table")

    table_cols = [
        "user_id",
        "assigned_pc",
        "last_seen_pc",
        "threat_score",
        "risk_level",
        "rf_threat_probability",
        "lof_anomaly_score",
        "after_hours_logons",
        "foreign_pc_logons",
        "foreign_pc_ratio",
        "usb_connects",
        "http_uploads",
        "external_emails",
        "copy_to_removable",
        "exfiltration_score",
        "suspicious_activity",
    ]
    table_cols = [c for c in filtered.columns if c in table_cols]

    st.dataframe(
        filtered[table_cols].head(max_rows),
        use_container_width=True,
        hide_index=True,
        height=450
    )

st.divider()
st.caption(f"Dashboard refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")