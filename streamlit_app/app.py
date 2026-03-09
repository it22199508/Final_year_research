import streamlit as st
from pathlib import Path
import sys, time
import pandas as pd
import plotly.express as px
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from threat_mapping import categorize_threats, summary_counts, PROFILES
from data_sources import list_csv_instances

st.set_page_config(page_title="HEADS • Threat Monitoring", layout="wide", page_icon="🛡️")

# Initialize dark mode in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ---------- Sidebar controls (no model folder / no directions) ----------
with st.sidebar:
    st.header("Data feed")

    feed_mode = st.selectbox(
        "Select data source",
        ["Live feed", "Test instance (dropdown)", "Default scored_events.csv"],
        index=0
    )

    # Threat mapping "testing models" dropdown (profiles)
    profile = st.selectbox("Threat mapping profile", list(PROFILES.keys()), index=1)

    min_prob = st.slider("Min probability (if available)", 0.0, 1.0, 0.50, 0.01)
    rows_to_show = st.slider("Rows to show", 50, 1000, 300, 50)

    st.divider()
    dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_mode
    auto_refresh = st.toggle("Auto refresh", value=False)
    refresh_sec = st.slider("Refresh interval (sec)", 2, 30, 5)
    refresh_now = st.button("🔄 Refresh")

# Dark dashboard CSS (closer to sample)
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

# Handle refresh without extra dependency
if refresh_now:
    st.rerun()
if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()

# ---------- Load data ----------
df = None
source_label = ""

if feed_mode == "Live feed":
    p = Path("data/live/live_events.csv")
    source_label = "LIVE"
    if p.exists():
        try:
            df = pd.read_csv(p)
        except Exception:
            df = None
    if df is None or len(df)==0:
        # fallback to default scored if live empty
        p2 = Path("data/processed/scored_events.csv")
        source_label = "LIVE (empty) → SCORDED"
        if not p2.exists():
            st.error("No live data and scored_events.csv not found. Run training notebook first.")
            st.stop()
        df = pd.read_csv(p2)

elif feed_mode == "Default scored_events.csv":
    p = Path("data/processed/scored_events.csv")
    source_label = "SCORDED (default)"
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
    for i,p in enumerate(files):
        tag = " (default)" if p.name == "scored_events.csv" else ""
        labels.append(p.name + tag)
        if p.name == "scored_events.csv":
            idx_default = i
    chosen = st.selectbox("Choose test instance", labels, index=idx_default)
    chosen_path = files[labels.index(chosen)]
    source_label = f"TEST: {chosen_path.name}"
    df = pd.read_csv(chosen_path)

# ---------- Threat mapping ----------
df = categorize_threats(df, profile=profile)

# probability filter if exists
if "xgb_proba" in df.columns:
    df["xgb_proba"] = pd.to_numeric(df["xgb_proba"], errors="coerce").fillna(0.0)
    if min_prob > 0:
        df = df[df["xgb_proba"] >= min_prob]

# Build views
alerts_df = df[df["threat_category"] != "Normal"].copy()


# ---------- Top header / KPIs ----------
st.markdown(
    f"""
    <div style='padding-top:0.5rem; padding-bottom:0.5rem; width:100%;'>
        <span style='font-size:2.2rem; font-weight:700; vertical-align:middle;'>Threat Monitoring</span>
        <span class='small' style='font-size:1.1rem; margin-left:0.5rem; vertical-align:middle;'>({source_label})</span>
    </div>
    """,
    unsafe_allow_html=True
)

total_users = df["src_ip"].nunique() if "src_ip" in df.columns else 0
alerts = len(alerts_df)
avg_score = float(df["xgb_proba"].mean()) if "xgb_proba" in df.columns and len(df)>0 else float(df.get("temporal_score", pd.Series([0.0])).mean())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Users Analysed", f"{total_users:,}")
k2.metric("High Risk", f"{int((alerts_df['threat_category']!='Normal').sum()):,}" if len(alerts_df) else "0")
k3.metric("Critical", f"{int((alerts_df['xgb_proba']>=0.9).sum()):,}" if "xgb_proba" in alerts_df.columns else "0")
k4.metric("Medium Risk", f"{int(((alerts_df.get('xgb_proba',0)>=0.6) & (alerts_df.get('xgb_proba',0)<0.9)).sum()):,}" if "xgb_proba" in alerts_df.columns else "0")
k5.metric("Average Score", f"{avg_score:.3f}")

# Risk level breakdown row (based on proba if available, else based on category only)
risk_levels = ["Very Low","Low","Medium","High","Critical"]
if "xgb_proba" in df.columns:
    p = df["xgb_proba"].fillna(0.0)
    risk = pd.cut(p, bins=[-1,0.2,0.4,0.6,0.8,1.0], labels=risk_levels)
else:
    risk = pd.Series(["Low"]*len(df))

r_counts = risk.value_counts().reindex(risk_levels, fill_value=0)
r1,r2,r3,r4,r5 = st.columns(5)
r1.metric("VERY LOW", int(r_counts["Very Low"]))
r2.metric("LOW", int(r_counts["Low"]))
r3.metric("MEDIUM", int(r_counts["Medium"]))
r4.metric("HIGH", int(r_counts["High"]))
r5.metric("CRITICAL", int(r_counts["Critical"]))

# ---------- Tabs like sample ----------
tab_overview, tab_live, tab_users, tab_eval, tab_table = st.tabs(["Overview", "Threat Preview", "Category Views", "Model Evaluation", "Analyst Table"])

with tab_overview:
    c1,c2,c3 = st.columns([1.1, 1.1, 1.8])

    with c1:
        summ = summary_counts(df, "threat_category")
        summ = summ[summ["threat_category"]!="Normal"]
        if len(summ)==0:
            st.markdown("<div class='card'><b>Threat Categories</b><br/><span class='small'>No alerts</span></div>", unsafe_allow_html=True)
        else:
            st.plotly_chart(px.pie(summ, names="threat_category", values="count", hole=0.65, title="Threat Categories"), use_container_width=True)

    with c2:
        # Show category counts bar
        summ2 = summary_counts(df, "threat_category")
        st.plotly_chart(px.bar(summ2, x="threat_category", y="count", title="Threat Category Count"), use_container_width=True)

    with c3:
        if "date" in df.columns:
            trend = df.groupby("date").size().reset_index(name="count")
            st.plotly_chart(px.area(trend, x="date", y="count", title="Logs Trend"), use_container_width=True)
        else:
            st.markdown("<div class='card'><b>Logs Trend</b><br/><span class='small'>No timestamp available</span></div>", unsafe_allow_html=True)

    st.divider()
    # 3 graphs for 3 categories
    g1,g2,g3 = st.columns(3)
    cats = ["Suspicious Login","Privilege Abuse","Account Takeover"]

    def cat_trend(cat):
        d = df[df["threat_category"]==cat]
        if "date" not in d.columns or len(d)==0:
            return None
        return d.groupby("date").size().reset_index(name="count")

    for col, cat in zip([g1,g2,g3], cats):
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
    cols_show = [c for c in ["timestamp","src_ip","dst_ip","protocol","src_port","dst_port","xgb_proba","threat_category","temporal_score","relational_score"] if c in view.columns]
    st.dataframe(view[cols_show].head(rows_to_show), use_container_width=True)

with tab_users:
    # Category-specific overviews
    cat = st.selectbox("Select category", ["Suspicious Login","Privilege Abuse","Account Takeover"])
    d = df[df["threat_category"]==cat].copy()
    st.markdown(f"#### {cat} Overview", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Events", f"{len(d):,}")
    c2.metric("Actors", d["src_ip"].nunique() if "src_ip" in d.columns else 0)
    c3.metric("Targets", d["dst_ip"].nunique() if "dst_ip" in d.columns else 0)

    colA,colB = st.columns(2)
    if "src_ip" in d.columns and len(d)>0:
        topa = d["src_ip"].astype(str).value_counts().head(10).reset_index()
        topa.columns = ["src_ip","count"]
        colA.plotly_chart(px.bar(topa, x="src_ip", y="count", title="Top Actors"), use_container_width=True)
    if "dst_ip" in d.columns and len(d)>0:
        topt = d["dst_ip"].astype(str).value_counts().head(10).reset_index()
        topt.columns = ["dst_ip","count"]
        colB.plotly_chart(px.bar(topt, x="dst_ip", y="count", title="Top Targets"), use_container_width=True)

    if "date" in d.columns and len(d)>0:
        tr = d.groupby("date").size().reset_index(name="count")
        st.plotly_chart(px.line(tr, x="date", y="count", title=f"{cat} trend"), use_container_width=True)

with tab_eval:
    st.markdown("<span class='small'>Evaluate scored test instances (data/test/) or the default scored_events.csv.</span>", unsafe_allow_html=True)
    from data_sources import list_csv_instances
    files = list_csv_instances()
    if not files:
        st.warning("No CSV instances found in data/test/ or data/processed/.")
    else:
        labels=[]
        idx=0
        for i,p in enumerate(files):
            tag = " (default)" if p.name=="scored_events.csv" else ""
            labels.append(p.parent.name + "/" + p.name + tag)
            if p.name=="scored_events.csv":
                idx=i
        chosen = st.selectbox("Evaluation dataset (scored CSV)", labels, index=idx, key="eval_select")
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="eval_thresh")
        p = files[labels.index(chosen)]
        dfe = pd.read_csv(p)
        if not {"label","xgb_proba"}.issubset(dfe.columns):
            st.error("Selected file must include label and xgb_proba.")
        else:
            y_true = pd.to_numeric(dfe["label"], errors="coerce").fillna(0).astype(int).to_numpy()
            y_score = pd.to_numeric(dfe["xgb_proba"], errors="coerce").fillna(0).astype(float).to_numpy()
            y_pred = (y_score >= threshold).astype(int)

            k1,k2,k3,k4 = st.columns(4)
            if len(np.unique(y_true))>1:
                try:
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    k1.metric("ROC-AUC", f"{roc_auc_score(y_true, y_score):.3f}")
                    k2.metric("PR-AUC", f"{average_precision_score(y_true, y_score):.3f}")
                except ImportError:
                    k1.metric("ROC-AUC", "Scikit-learn Error")
                    k2.metric("PR-AUC", "Scikit-learn Error")
            else:
                k1.metric("ROC-AUC","N/A"); k2.metric("PR-AUC","N/A")
            k3.metric("Attack rate", f"{(y_true==1).mean()*100:.2f}%")
            k4.metric("Predicted positive", f"{(y_pred==1).mean()*100:.2f}%")

            st.plotly_chart(px.histogram(dfe, x="xgb_proba", nbins=60, title="Probability distribution"), use_container_width=True)

            if len(np.unique(y_true))>1:
                try:
                    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
                    import plotly.graph_objects as go
                    cm = confusion_matrix(y_true, y_pred)
                    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
                    fig = go.Figure(data=go.Heatmap(z=cm_norm, x=["Pred 0","Pred 1"], y=["True 0","True 1"]))
                    fig.update_layout(title="Confusion Matrix (row-normalized)", height=320)
                    st.plotly_chart(fig, use_container_width=True)

                    fpr,tpr,_=roc_curve(y_true,y_score)
                    st.plotly_chart(px.line(pd.DataFrame({"fpr":fpr,"tpr":tpr}), x="fpr", y="tpr", title="ROC Curve"), use_container_width=True)
                    prec,rec,_=precision_recall_curve(y_true,y_score)
                    st.plotly_chart(px.line(pd.DataFrame({"recall":rec,"precision":prec}), x="recall", y="precision", title="PR Curve"), use_container_width=True)
                except ImportError:
                    st.error("Scikit-learn is not available. Please check the installation.")
            else:
                st.info("Need both classes in evaluation set to draw ROC/PR curves.")

with tab_table:
    st.markdown("<span class='small'>Analyst view: filter by risk level and minimum score.</span>", unsafe_allow_html=True)
    # Filters
    min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
    risk_sel = st.selectbox("Risk level", ["All"] + risk_levels)

    tdf = df.copy()
    if "xgb_proba" in tdf.columns:
        tdf = tdf[tdf["xgb_proba"].fillna(0.0) >= min_score]
    if risk_sel != "All":
        tdf = tdf[risk == risk_sel]

    cols_show = [c for c in ["timestamp","src_ip","dst_ip","protocol","src_port","dst_port","xgb_proba","threat_category"] if c in tdf.columns]
    if "timestamp" in tdf.columns:
        tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
        tdf = tdf.sort_values("timestamp", ascending=False)
    st.dataframe(tdf[cols_show].head(rows_to_show), use_container_width=True)
