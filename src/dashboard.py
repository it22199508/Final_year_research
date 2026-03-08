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

st.title("🚨 NSADM Insider Threat Monitoring Dashboard")
st.caption("CERT r4.2 | Unseen Test Users | Clear Analyst View")

# -------------------------
# LOAD DATA
# -------------------------
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
                    return df.sort_values("timestamp", ascending=False)
        except Exception:
            pass
    return pd.DataFrame()

def load_test_dashboard():
    if os.path.exists(TEST_DASHBOARD_FILE):
        df = pd.read_csv(TEST_DASHBOARD_FILE)
        if "threat_score" in df.columns:
            df["threat_score"] = pd.to_numeric(df["threat_score"], errors="coerce").fillna(0)
        return df.sort_values("threat_score", ascending=False)
    return pd.DataFrame()

live_df = load_live_alerts()
full_df = load_test_dashboard()

if full_df.empty:
    st.error("Test dashboard data not found. Run predict_test_users.py first.")
    st.stop()

# -------------------------
# TOP KPI SECTION
# -------------------------
high_count = int((full_df["threat_score"] >= THREAT_THRESHOLD).sum())
critical_count = int((full_df["threat_score"] >= CRITICAL_THRESHOLD).sum())
medium_count = int(((full_df["threat_score"] >= 0.5) & (full_df["threat_score"] < THREAT_THRESHOLD)).sum())
low_count = int(((full_df["threat_score"] >= 0.3) & (full_df["threat_score"] < 0.5)).sum())
very_low_count = int((full_df["threat_score"] < 0.3).sum())
avg_score = float(full_df["threat_score"].mean()) if len(full_df) else 0.0

last_alert_time = "N/A"
if not live_df.empty and "timestamp" in live_df.columns and live_df["timestamp"].notna().any():
    last_alert_time = live_df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Users Analysed", len(full_df))
c2.metric("High Risk", high_count)
c3.metric("Critical", critical_count)
c4.metric("Medium Risk", medium_count)
c5.metric("Average Score", f"{avg_score:.3f}")

st.divider()

# -------------------------
# RISK BREAKDOWN CARDS
# -------------------------
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("VERY LOW", very_low_count)
r2.metric("LOW", low_count)
r3.metric("MEDIUM", medium_count)
r4.metric("HIGH", high_count)
r5.metric("CRITICAL", critical_count)

st.divider()

# -------------------------
# FILTERS
# -------------------------
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

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Live Alerts", "All Test Users", "Analyst Table"])

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
    st.subheader("Latest Live Alerts (High / Critical Only)")

    if live_df.empty:
        st.info("No live alerts yet. Start realtime_monitor.py")
    else:
        latest_alerts = live_df.head(max_rows)

        for _, row in latest_alerts.iterrows():
            score = float(row["threat_score"])
            risk = str(row.get("risk_level", "UNKNOWN"))
            user_id = str(row.get("user_id", "UNKNOWN"))
            ts = row.get("timestamp", None)
            ts_text = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(ts) else "N/A"

            if score >= CRITICAL_THRESHOLD:
                badge = "🔴 CRITICAL"
                color = "#dc2626"
            else:
                badge = "🟠 HIGH"
                color = "#f97316"

            a, b = st.columns([4, 1])

            with a:
                st.markdown(f"### {user_id}")
                st.markdown(f"**Time:** {ts_text}")
                st.markdown(f"**Threat Score:** {score:.3f}")
                st.markdown(f"**Risk Level:** {risk}")

                indicators = []
                if row.get("after_hours_logons", 0) > 0:
                    indicators.append(f"🌙 {int(row['after_hours_logons'])} after-hours logons")
                if row.get("foreign_pc_logons", 0) > 0:
                    indicators.append(f"🖥️ {int(row['foreign_pc_logons'])} foreign-PC logons")
                if row.get("usb_connects", 0) > 0:
                    indicators.append(f"🔌 {int(row['usb_connects'])} USB connects")
                if row.get("external_emails", 0) > 0:
                    indicators.append(f"📧 {int(row['external_emails'])} external emails")
                if row.get("http_uploads", 0) > 0:
                    indicators.append(f"⬆️ {int(row['http_uploads'])} upload events")
                if row.get("copy_to_removable", 0) > 0:
                    indicators.append(f"💾 {int(row['copy_to_removable'])} removable-media copies")

                if indicators:
                    st.markdown(" | ".join(indicators))

            with b:
                st.markdown(f"### {badge}")
                st.progress(min(max(score, 0.0), 1.0), text=f"{score:.1%}")

            st.markdown(
                f'<div style="height:4px; background-color:{color}; border-radius:8px; margin-bottom:16px;"></div>',
                unsafe_allow_html=True
            )

with tab3:
    st.subheader("All Test Users (Includes Low / Medium / High)")

    summary_cols = [
        "user_id",
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
    table_cols = [c for c in table_cols if c in filtered.columns]

    st.dataframe(
        filtered[table_cols].head(max_rows),
        use_container_width=True,
        hide_index=True,
        height=450
    )

st.divider()
st.caption(f"Last dashboard refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")