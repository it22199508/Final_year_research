from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover
    st_autorefresh = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None


st.set_page_config(
    page_title="AI-Driven Insider Threat SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass(frozen=True)
class RiskWeights:
    behavior: float = 0.35
    access: float = 0.25
    device_risk: float = 0.20
    time: float = 0.20


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _risk_level(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def _risk_color(level: str) -> str:
    return {"Low": "#27F3C2", "Medium": "#FF9A3D", "High": "#FF3355"}.get(level, "#9CA3AF")


def _org_risk_score(scores: np.ndarray) -> float:
    if scores.size == 0:
        return 0.0
    top = np.sort(scores)[-max(10, int(0.04 * scores.size)) :]
    return float(np.clip(0.65 * scores.mean() + 0.35 * top.mean(), 0, 100))


def _inject_css() -> None:
    st.markdown(
        """
<style>
    /* Base */
    .stApp { background: radial-gradient(1200px 800px at 25% 20%, rgba(0, 200, 255, 0.10), rgba(0,0,0,0) 55%),
                         radial-gradient(1200px 800px at 85% 10%, rgba(255, 51, 85, 0.10), rgba(0,0,0,0) 60%),
                         linear-gradient(180deg, #070A12 0%, #070A12 40%, #060813 100%);
            color: rgba(255,255,255,0.92);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.03) 100%);
        border-right: 1px solid rgba(0, 200, 255, 0.18);
        backdrop-filter: blur(10px);
    }

    /* Cards / Glass */
    .soc-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(0, 200, 255, 0.18);
        border-radius: 16px;
        padding: 14px 14px 12px 14px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
    }
    .soc-subtitle {
        font-size: 0.85rem;
        opacity: 0.78;
        margin-top: -6px;
        margin-bottom: 12px;
    }
    .chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        border: 1px solid rgba(255,255,255,0.16);
        background: rgba(255,255,255,0.06);
    }
    .chip-high { border-color: rgba(255, 51, 85, 0.55); box-shadow: 0 0 0 1px rgba(255, 51, 85, 0.18) inset; }
    .chip-med  { border-color: rgba(255, 154, 61, 0.55); box-shadow: 0 0 0 1px rgba(255, 154, 61, 0.18) inset; }
    .chip-low  { border-color: rgba(39, 243, 194, 0.55); box-shadow: 0 0 0 1px rgba(39, 243, 194, 0.18) inset; }

    /* Metric tweaks */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(0, 200, 255, 0.14);
        padding: 12px 12px;
        border-radius: 14px;
    }
    div[data-testid="stMetric"] label { opacity: 0.80; }

    /* Alert banner */
    .soc-alert {
        border-radius: 14px;
        padding: 10px 12px;
        border: 1px solid rgba(255, 51, 85, 0.35);
        background: linear-gradient(90deg, rgba(255, 51, 85, 0.14), rgba(255, 51, 85, 0.04));
    }
    .soc-blue {
        border: 1px solid rgba(0, 200, 255, 0.25);
        background: linear-gradient(90deg, rgba(0, 200, 255, 0.14), rgba(0, 200, 255, 0.04));
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        overflow: hidden;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    if "soc_users" not in st.session_state:
        rng = np.random.default_rng(2026)
        n_users = 240
        depts = np.array(["Finance", "IT", "HR", "Clinical", "Operations", "Security", "R&D"])
        dept = rng.choice(depts, size=n_users, p=[0.18, 0.16, 0.11, 0.14, 0.18, 0.10, 0.13])
        remote_ratio = np.clip(rng.normal(loc=0.55, scale=0.25, size=n_users), 0, 1)
        user_id = np.array([f"U-{i:05d}" for i in range(1, n_users + 1)])
        st.session_state["soc_users"] = pd.DataFrame(
            {
                "user_id": user_id,
                "department": dept,
                "remote_ratio": remote_ratio,
            }
        )

    st.session_state.setdefault("soc_trend", [])
    st.session_state.setdefault("soc_alerts", [])
    st.session_state.setdefault("soc_audit", [])
    st.session_state.setdefault("soc_selected_user", None)


def _simulate_snapshot(tick: int, weights: RiskWeights) -> tuple[pd.DataFrame, dict]:
    users = st.session_state["soc_users"].copy()
    rng = np.random.default_rng(1200 + int(tick))

    base_behavior = rng.normal(42, 18, size=len(users))
    base_access = rng.normal(38, 18, size=len(users))
    base_time = rng.normal(35, 20, size=len(users))
    base_device_trust = rng.normal(78, 12, size=len(users))

    remote_bias = (users["remote_ratio"].to_numpy() - 0.5) * 18
    users["behavior_anomaly"] = np.clip(base_behavior + remote_bias + rng.normal(0, 8, size=len(users)), 0, 100)
    users["access_anomaly"] = np.clip(base_access + rng.normal(0, 10, size=len(users)), 0, 100)
    users["time_anomaly"] = np.clip(base_time + (remote_bias * 0.7) + rng.normal(0, 10, size=len(users)), 0, 100)
    users["device_trust"] = np.clip(base_device_trust - (remote_bias * 0.45) + rng.normal(0, 6, size=len(users)), 0, 100)

    device_risk = 100.0 - users["device_trust"].to_numpy()
    risk = (
        weights.behavior * users["behavior_anomaly"].to_numpy()
        + weights.access * users["access_anomaly"].to_numpy()
        + weights.device_risk * device_risk
        + weights.time * users["time_anomaly"].to_numpy()
    )
    users["risk_score"] = np.clip(risk, 0, 100)
    users["risk_level"] = users["risk_score"].map(_risk_level)

    last_activity = pd.to_datetime(_now_utc())
    users["last_activity"] = (
        last_activity - pd.to_timedelta(rng.integers(1, 180, size=len(users)), unit="m")
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    scores = users["risk_score"].to_numpy()
    summary = {
        "total_users": int(len(users)),
        "high": int((users["risk_level"] == "High").sum()),
        "medium": int((users["risk_level"] == "Medium").sum()),
        "low": int((users["risk_level"] == "Low").sum()),
        "org_score": _org_risk_score(scores),
    }
    return users, summary


def _append_trend(org_score: float) -> None:
    trend = st.session_state["soc_trend"]
    trend.append({"ts": _now_utc(), "org_risk": float(org_score)})
    st.session_state["soc_trend"] = trend[-60:]


def _emit_alerts(users: pd.DataFrame, tick: int) -> None:
    critical = users.sort_values("risk_score", ascending=False).head(4)
    critical = critical[critical["risk_score"] >= 88]
    if critical.empty:
        return

    alerts = st.session_state["soc_alerts"]
    for _, row in critical.iterrows():
        alert_id = f"{tick}:{row['user_id']}"
        message = f"CRITICAL: {row['user_id']} risk={row['risk_score']:.0f} ({row['department']})"
        alerts.append(
            {
                "id": alert_id,
                "ts": _now_utc().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "severity": "Critical",
                "message": message,
                "user_id": row["user_id"],
                "status": "Open",
            }
        )

    st.session_state["soc_alerts"] = alerts[-30:]
    newest = alerts[-1]
    try:
        st.toast(newest["message"], icon="🚨")
    except Exception:
        # toast isn't critical; fall back to visible list only
        pass


def _audit(action: str, user_id: str | None, details: str) -> None:
    audit = st.session_state["soc_audit"]
    audit.append(
        {
            "ts": _now_utc().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "user_id": user_id or "-",
            "action": action,
            "details": details,
        }
    )
    st.session_state["soc_audit"] = audit[-80:]


def _require_plotly() -> bool:
    if px is None or go is None:
        st.error("Missing optional dependency: plotly. Install requirements.txt to enable charts.")
        return False
    return True


def _executive_overview(users: pd.DataFrame, summary: dict) -> None:
    st.markdown("### Executive Overview Panel")
    st.markdown('<div class="soc-subtitle">Hybrid work insider threat posture • live simulation</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1.1, 1, 1, 1, 1.25])
    c1.metric("Total monitored users", summary["total_users"])
    c2.metric("High risk", summary["high"])
    c3.metric("Medium risk", summary["medium"])
    c4.metric("Low risk", summary["low"])
    c5.metric("Org risk score", f"{summary['org_score']:.1f} / 100")

    st.markdown("#### Live risk trend graph")
    if not _require_plotly():
        return

    trend = pd.DataFrame(st.session_state["soc_trend"])
    if trend.empty:
        st.info("Trend warming up…")
        return

    fig = px.line(
        trend,
        x="ts",
        y="org_risk",
        markers=True,
        title=None,
    )
    fig.update_traces(line=dict(color="#00C8FF", width=3), marker=dict(size=6, color="#00C8FF"))
    fig.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 100], title="Risk"),
        xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)


def _ai_risk_engine(users: pd.DataFrame, weights: RiskWeights) -> None:
    st.markdown("### AI Risk Scoring Engine Visualization")
    st.markdown(
        '<div class="soc-subtitle">Dynamic 0–100 score • weighted factors • risk class thresholds</div>',
        unsafe_allow_html=True,
    )

    selected = st.session_state.get("soc_selected_user")
    default_user = users.sort_values("risk_score", ascending=False).iloc[0]["user_id"]
    user_id = st.selectbox("User", users["user_id"], index=int(users.index[users["user_id"] == (selected or default_user)][0]))
    st.session_state["soc_selected_user"] = user_id

    row = users.loc[users["user_id"] == user_id].iloc[0]
    score = float(row["risk_score"])
    level = _risk_level(score)
    chip_class = {"High": "chip-high", "Medium": "chip-med", "Low": "chip-low"}[level]

    st.markdown(
        f'<div class="soc-card">'
        f'<span class="chip {chip_class}">Risk level: {level}</span>'
        f'<span style="margin-left:12px; opacity:0.85">Dynamic risk score</span>'
        f'<div style="font-size:2.2rem; font-weight:800; margin-top:6px; color:{_risk_color(level)}">{score:.0f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Weighted risk calculation factors")
    factors = pd.DataFrame(
        {
            "factor": [
                "Behavior anomaly score",
                "Access anomaly score",
                "Device trust score",
                "Time-based anomaly score",
            ],
            "value": [
                float(row["behavior_anomaly"]),
                float(row["access_anomaly"]),
                float(row["device_trust"]),
                float(row["time_anomaly"]),
            ],
            "weight": [weights.behavior, weights.access, weights.device_risk, weights.time],
            "direction": ["↑ increases risk", "↑ increases risk", "↓ decreases risk", "↑ increases risk"],
        }
    )

    if _require_plotly():
        display = factors.copy()
        display["impact"] = display.apply(
            lambda r: r["weight"] * (100 - r["value"]) if r["factor"] == "Device trust score" else r["weight"] * r["value"],
            axis=1,
        )
        fig = px.bar(
            display,
            x="impact",
            y="factor",
            orientation="h",
            hover_data={"weight": ":.2f", "value": ":.1f", "direction": True, "impact": ":.2f"},
            title=None,
        )
        fig.update_traces(marker_color="#00C8FF")
        fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Weighted contribution (simulated)",
            yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(factors, use_container_width=True)


def _explainable_ai(users: pd.DataFrame, weights: RiskWeights) -> None:
    st.markdown("### Explainable AI Section")
    st.markdown(
        '<div class="soc-subtitle">SHAP/LIME-style feature importance • top drivers • confidence indicator</div>',
        unsafe_allow_html=True,
    )

    if not _require_plotly():
        return

    user_id = st.session_state.get("soc_selected_user") or users.sort_values("risk_score", ascending=False).iloc[0]["user_id"]
    row = users.loc[users["user_id"] == user_id].iloc[0]
    score = float(row["risk_score"])
    level = _risk_level(score)

    # Simulated local explanation contributions
    contributions = {
        "Behavior anomaly": weights.behavior * float(row["behavior_anomaly"]),
        "Access anomaly": weights.access * float(row["access_anomaly"]),
        "Device risk": weights.device_risk * float(100 - row["device_trust"]),
        "Time anomaly": weights.time * float(row["time_anomaly"]),
        "Dept sensitivity": float({"Finance": 6.2, "Clinical": 7.2, "Security": 5.8}.get(row["department"], 3.5)),
    }

    imp = (
        pd.DataFrame({"feature": list(contributions.keys()), "importance": list(contributions.values())})
        .sort_values("importance", ascending=False)
        .head(5)
    )

    hover = {
        "Behavior anomaly": "Unusual sequences vs baseline behavior profile",
        "Access anomaly": "Privilege escalation / atypical resource access",
        "Device risk": "Lower device trust increases risk contribution",
        "Time anomaly": "Off-hours activity / abnormal login time patterns",
        "Dept sensitivity": "Higher sensitivity departments amplify review priority",
    }
    imp["explain"] = imp["feature"].map(hover)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.bar(
            imp.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title=None,
            hover_data={"explain": True, "importance": ":.2f"},
        )
        fig.update_traces(marker_color="#FF9A3D" if level == "Medium" else ("#FF3355" if level == "High" else "#27F3C2"))
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Feature impact (simulated)",
            yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Confidence is a bounded function of score; higher risk often yields sharper separation.
        conf = float(np.clip(0.55 + (abs(score - 50) / 100) * 0.45, 0.50, 0.95))
        st.markdown(
            f'<div class="soc-card">'
            f'<div style="opacity:0.85">Model confidence indicator</div>'
            f'<div style="font-size:1.8rem; font-weight:800; margin-top:6px">{conf*100:.1f}%</div>'
            f'<div style="opacity:0.72; margin-top:6px">Higher means stronger separation under the current simulated conditions.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(conf)

    st.markdown("#### Tooltip explanations")
    st.info("Hover the bars in the feature chart to see impact explanations.")


def _hybrid_work_monitoring(users: pd.DataFrame) -> None:
    st.markdown("### Hybrid Work Monitoring Panel")
    st.markdown(
        '<div class="soc-subtitle">Remote vs onsite comparison • abnormal logins • device + geo anomalies</div>',
        unsafe_allow_html=True,
    )

    if not _require_plotly():
        return

    remote = users[users["remote_ratio"] >= 0.60]
    onsite = users[users["remote_ratio"] <= 0.40]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Remote vs Onsite behavior comparison")
        comp = pd.DataFrame(
            {
                "group": ["Remote", "Onsite"],
                "behavior_anomaly": [remote["behavior_anomaly"].mean(), onsite["behavior_anomaly"].mean()],
                "access_anomaly": [remote["access_anomaly"].mean(), onsite["access_anomaly"].mean()],
                "time_anomaly": [remote["time_anomaly"].mean(), onsite["time_anomaly"].mean()],
                "device_trust": [remote["device_trust"].mean(), onsite["device_trust"].mean()],
            }
        )
        melt = comp.melt(id_vars=["group"], var_name="signal", value_name="avg")
        fig = px.bar(
            melt,
            x="signal",
            y="avg",
            color="group",
            barmode="group",
            hover_data={"avg": ":.1f"},
        )
        fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Abnormal login time detection visualization")
        rng = np.random.default_rng(900 + int(st.session_state.get("soc_tick", 0)))
        hours = np.arange(24)
        remote_counts = np.clip(rng.poisson(lam=7 + (hours < 6) * 6 + (hours > 20) * 7, size=24), 0, None)
        onsite_counts = np.clip(rng.poisson(lam=8 + (hours < 6) * 2 + (hours > 20) * 2, size=24), 0, None)
        heat = np.vstack([remote_counts, onsite_counts])
        fig = go.Figure(
            data=go.Heatmap(
                z=heat,
                x=hours,
                y=["Remote", "Onsite"],
                colorscale=[
                    [0.0, "#0B1222"],
                    [0.45, "#00C8FF"],
                    [0.75, "#FF9A3D"],
                    [1.0, "#FF3355"],
                ],
                hovertemplate="%{y} • Hour %{x}: %{z} abnormal logins<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Hour of day",
            yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Suspicious device usage tracking")
    suspicious = users.sort_values("device_trust").head(8)[["user_id", "department", "device_trust", "risk_score"]]
    suspicious = suspicious.assign(device_trust=lambda d: d["device_trust"].map(lambda x: f"{x:.1f}"))
    st.dataframe(suspicious, use_container_width=True, hide_index=True)

    st.markdown("#### Geolocation anomaly alerts")
    rng = np.random.default_rng(333 + int(st.session_state.get("soc_tick", 0)))
    geo_pool = [
        ("US", "New York"),
        ("US", "Chicago"),
        ("GB", "London"),
        ("DE", "Frankfurt"),
        ("AE", "Dubai"),
        ("SG", "Singapore"),
        ("BR", "São Paulo"),
        ("JP", "Tokyo"),
        ("ZA", "Johannesburg"),
    ]
    sample_users = users.sort_values("risk_score", ascending=False).head(18).sample(6, random_state=int(rng.integers(0, 1_000_000)))
    rows = []
    for u in sample_users["user_id"].to_list():
        cc, city = geo_pool[int(rng.integers(0, len(geo_pool)))]
        rows.append(
            {
                "ts": _now_utc().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "user_id": u,
                "country": cc,
                "city": city,
                "alert": "Geo-velocity anomaly (simulated)",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _high_risk_investigation(users: pd.DataFrame) -> None:
    st.markdown("### High-Risk User Investigation Panel")
    st.markdown(
        '<div class="soc-subtitle">Flagged users • investigation workflow • mitigation audit trail</div>',
        unsafe_allow_html=True,
    )

    flagged = users[users["risk_score"] >= 70].sort_values("risk_score", ascending=False).copy()
    flagged = flagged[["user_id", "department", "risk_score", "risk_level", "last_activity"]]

    if flagged.empty:
        st.info("No high-risk users right now. Live simulation may generate critical cases on next refresh.")
        return

    st.markdown("#### Table of flagged users")
    header = st.columns([1.1, 1.3, 1.0, 1.0, 1.8, 1.1])
    header[0].markdown("**User ID**")
    header[1].markdown("**Department**")
    header[2].markdown("**Risk Score**")
    header[3].markdown("**Risk Level**")
    header[4].markdown("**Last Activity**")
    header[5].markdown("**Action**")

    for _, row in flagged.head(15).iterrows():
        cols = st.columns([1.1, 1.3, 1.0, 1.0, 1.8, 1.1])
        cols[0].write(row["user_id"])
        cols[1].write(row["department"])
        cols[2].write(f"{row['risk_score']:.0f}")
        cols[3].markdown(
            f"<span class='chip {('chip-high' if row['risk_level']=='High' else 'chip-med')}'>{row['risk_level']}</span>",
            unsafe_allow_html=True,
        )
        cols[4].write(row["last_activity"])
        if cols[5].button("Investigate", key=f"inv_{row['user_id']}"):
            st.session_state["soc_selected_user"] = row["user_id"]
            _audit("Investigate", row["user_id"], "Opened investigation panel")

    st.markdown("#### Simulated mitigation action log (audit trail)")
    audit = pd.DataFrame(st.session_state["soc_audit"])
    if audit.empty:
        st.info("Audit trail will populate as you Investigate or run response actions.")
    else:
        st.dataframe(audit.tail(20), use_container_width=True, hide_index=True)


def _alerts_incident_response(users: pd.DataFrame) -> None:
    st.markdown("### Alert & Incident Response Section")
    st.markdown(
        '<div class="soc-subtitle">Real-time critical alerts • severity indicator • automated response simulation</div>',
        unsafe_allow_html=True,
    )

    alerts = pd.DataFrame(st.session_state["soc_alerts"])
    if alerts.empty:
        st.markdown('<div class="soc-card soc-blue">No critical alerts at the moment.</div>', unsafe_allow_html=True)
        return

    open_alerts = alerts[alerts["status"] == "Open"].copy() if "status" in alerts.columns else alerts.copy()
    st.markdown("#### Real-time critical alerts popup")
    st.markdown(
        '<div class="soc-alert">New critical alerts also appear as toast notifications during live refresh.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Threat severity indicator")
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Open critical alerts", int((open_alerts["severity"] == "Critical").sum()))
    c2.metric("Total alerts", int(len(alerts)))
    worst_user = (
        users.sort_values("risk_score", ascending=False).iloc[0]["user_id"]
        if not users.empty
        else "-"
    )
    c3.metric("Highest-risk user", worst_user)

    st.markdown("#### Alerts")
    st.dataframe(open_alerts.tail(12), use_container_width=True, hide_index=True)

    st.markdown("#### Automated response simulation")
    user_id = st.session_state.get("soc_selected_user")
    if not user_id:
        user_id = open_alerts.iloc[-1]["user_id"]
        st.session_state["soc_selected_user"] = user_id

    r1, r2, r3 = st.columns([1.2, 1.2, 1.6])
    with r1:
        if st.button("Simulate account containment", key="resp_contain"):
            _audit("Response", user_id, "Containment simulated: session revoked + MFA reset")
            st.success("Containment simulated.")
    with r2:
        if st.button("Simulate device quarantine", key="resp_quarantine"):
            _audit("Response", user_id, "Quarantine simulated: device isolated from corp network")
            st.success("Device quarantine simulated.")
    with r3:
        st.caption("Actions write to the audit trail for research-ready screenshots.")


def main() -> None:
    _inject_css()
    _init_state()

    st.sidebar.markdown("## SOC Navigation")
    section = st.sidebar.radio(
        "Navigation",
        [
            "Executive Overview",
            "AI Risk Scoring Engine",
            "Explainable AI",
            "Hybrid Work Monitoring",
            "High-Risk Investigation",
            "Alerts & Incident Response",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Live simulation")
    live_mode = st.sidebar.toggle("Real-time updates", value=True)
    refresh_ms = st.sidebar.slider("Refresh interval (ms)", 1000, 8000, 2500, 250)
    if live_mode:
        if st_autorefresh is None:
            st.sidebar.warning("Install streamlit-autorefresh to enable auto-refresh.")
            tick = int(st.session_state.get("soc_tick", 0))
        else:
            tick = int(st_autorefresh(interval=refresh_ms, key="soc_autorefresh"))
        st.session_state["soc_tick"] = tick
    else:
        tick = int(st.session_state.get("soc_tick", 0))

    weights = RiskWeights()
    users, summary = _simulate_snapshot(tick=tick, weights=weights)
    _append_trend(summary["org_score"])
    if live_mode:
        _emit_alerts(users, tick=tick)

    st.markdown(
        "# AI-Driven Insider Threat & Anomaly Detection (Hybrid Work SOC)"
    )
    st.markdown(
        '<div class="soc-subtitle">Modern enterprise cybersecurity dashboard • banking/healthcare SOC style • research-ready</div>',
        unsafe_allow_html=True,
    )

    if section == "Executive Overview":
        _executive_overview(users, summary)
    elif section == "AI Risk Scoring Engine":
        _ai_risk_engine(users, weights)
    elif section == "Explainable AI":
        _explainable_ai(users, weights)
    elif section == "Hybrid Work Monitoring":
        _hybrid_work_monitoring(users)
    elif section == "High-Risk Investigation":
        _high_risk_investigation(users)
    elif section == "Alerts & Incident Response":
        _alerts_incident_response(users)
    else:
        st.error("Unknown section")


if __name__ == "__main__":
    main()
