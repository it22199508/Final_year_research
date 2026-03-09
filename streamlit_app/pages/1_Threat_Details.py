import streamlit as st
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import pandas as pd
from pathlib import Path
import plotly.express as px
from threat_mapping import categorize_threats, summary_counts, PROFILES

st.set_page_config(page_title="Threat Details • HEADS", layout="wide", page_icon="🚨")
st.title("Threat Details (Threat Categories Only)")

with st.sidebar:
    scored_path = st.text_input("Scored CSV", value="data/processed/scored_events.csv")
    profile = st.selectbox("Threat mapping profile", list(PROFILES.keys()), index=1)
    category = st.selectbox("Threat Category", [
        "Suspicious Login",
        "Privilege Abuse",
        "Account Takeover",
    ])

p = Path(scored_path)
if not p.exists():
    st.error("Scored CSV not found.")
    st.stop()

df = pd.read_csv(p)
df = categorize_threats(df, profile=profile)
df = df[df["threat_category"] == category]

c1, c2, c3 = st.columns(3)
c1.metric("Events", f"{len(df):,}")
c2.metric("Actors", df["src_ip"].nunique() if "src_ip" in df.columns else 0)
c3.metric("Targets", df["dst_ip"].nunique() if "dst_ip" in df.columns else 0)

colA, colB = st.columns(2)
if "src_ip" in df.columns and len(df)>0:
    topa = summary_counts(df, "src_ip").head(10)
    colA.plotly_chart(px.bar(topa, x="src_ip", y="count", title="Top Actors"), use_container_width=True)
if "dst_ip" in df.columns and len(df)>0:
    topt = summary_counts(df, "dst_ip").head(10)
    colB.plotly_chart(px.bar(topt, x="dst_ip", y="count", title="Top Targets"), use_container_width=True)

cols_show = [c for c in [
    "timestamp","src_ip","dst_ip","protocol","src_port","dst_port",
    "temporal_score","relational_score","xgb_proba","threat_category"
] if c in df.columns]

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", ascending=False)

st.dataframe(df[cols_show].head(500), use_container_width=True)
