from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from .config import HEADSConfig
from .features import add_derived_features, TabularFeaturizer
from .temporal_ae import TransformerAutoencoder, build_sequences, score_autoencoder
from .gnn import GraphSAGEEncoder, edge_anomaly_scores
from .graph import build_graph

app = FastAPI(title="HEADS Anomaly Detection API")

class Event(BaseModel):
    timestamp: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    bytes_sent: int
    bytes_received: int
    user_agent: str
    url: str = ""
    is_internal_traffic: bool = False

state = {"ready": False}

def load_models(model_dir: Path, device: str = "cpu"):
    feat = joblib.load(model_dir / "tabular_featurizer.joblib")

    # Transformer AE (weights only; needs feat_dim known at runtime)
    # We reconstruct with known feature list used in train_all.
    ae_cols = ["src_port","dst_port","bytes_sent","bytes_received","bytes_total","hour","dow","is_web","is_internal_traffic"]
    # load a dummy to get dimensions
    dummy = pd.DataFrame([{c:0 for c in ae_cols}])
    dummy["timestamp"] = pd.to_datetime(["2025-01-01"])
    dummy["src_ip"] = "0.0.0.0"
    dummy["dst_ip"] = "0.0.0.0"
    dummy["protocol"]="TCP"
    dummy["user_agent"]="UNK"
    dummy["url"]=""
    dummy["is_internal_traffic"]=0
    dummy = add_derived_features(dummy)
    feat_dim = len(ae_cols)

    ae = TransformerAutoencoder(feat_dim=feat_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1).to(device)
    ae.load_state_dict(torch.load(model_dir / "transformer_ae.pt", map_location=device))
    ae.eval()

    gnn = GraphSAGEEncoder(in_dim=2, hidden=64, layers=2).to(device)
    gnn.load_state_dict(torch.load(model_dir / "graphsage.pt", map_location=device))
    gnn.eval()

    # XGBoost
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(model_dir / "xgb.json"))

    return feat, ae, gnn, xgb_model

@app.on_event("startup")
def _startup():
    # Lazy: require user to pass --model-dir when starting as __main__
    pass

@app.post("/score")
def score_event(e: Event):
    if not state["ready"]:
        return {"error": "Models not loaded. Start with: python -m src.heads.serve_api --model-dir models --data-snapshot data/raw/cybersecurity.csv"}

    feat: TabularFeaturizer = state["feat"]
    ae = state["ae"]
    gnn = state["gnn"]
    xgb_model = state["xgb"]
    device = state["device"]

    # Build df for single event
    df = pd.DataFrame([e.model_dump()])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = add_derived_features(df)
    df["bytes_total"] = (df["bytes_sent"] + df["bytes_received"]).astype(int)
    df["is_internal_traffic"] = df["is_internal_traffic"].astype(int)

    # Temporal score for single event: we approximate by making a seq_len window from history snapshot
    hist = state["history"]
    actor_hist = hist[hist["src_ip"] == e.src_ip].copy()
    actor_hist = pd.concat([actor_hist, df], ignore_index=True).sort_values("timestamp")
    ae_cols = ["src_port","dst_port","bytes_sent","bytes_received","bytes_total","hour","dow","is_web","is_internal_traffic"]
    if len(actor_hist) >= state["seq_len"]:
        window = actor_hist.tail(state["seq_len"])[ae_cols].to_numpy(dtype=float)[None, :, :]
        temporal_score = float(score_autoencoder(ae, window, device=device)[0])
    else:
        temporal_score = 0.0

    # Relational score: rebuild graph from snapshot (simple, for demo)
    g_art = state["graph_artifacts"]
    import torch as _torch
    # unknown nodes fallback: score 0.0
    key_a = ("actor", e.src_ip)
    key_b = ("resource", e.dst_ip)
    if key_a in g_art.node_id_map and key_b in g_art.node_id_map:
        a = g_art.node_id_map[key_a]; b = g_art.node_id_map[key_b]
        edge_index = _torch.tensor([[a],[b]], dtype=_torch.long)
        relational_score = float(edge_anomaly_scores(gnn, g_art.data, edge_index, device=device)[0])
    else:
        relational_score = 0.0

    X_tab = feat.transform(df)
    X = np.hstack([X_tab, np.array([[temporal_score, relational_score]], dtype=float)])
    proba = float(xgb_model.predict_proba(X)[0,1])
    pred = int(proba >= 0.5)

    return {
        "temporal_score": temporal_score,
        "relational_score": relational_score,
        "xgb_proba": proba,
        "prediction": pred
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-snapshot", required=True, help="A CSV snapshot used to build graph + history for scoring")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seq-len", type=int, default=20)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    feat, ae, gnn, xgb_model = load_models(model_dir, device=args.device)

    hist = pd.read_csv(args.data_snapshot)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist = add_derived_features(hist)
    hist["bytes_total"] = (hist["bytes_sent"] + hist["bytes_received"]).astype(int)
    hist["is_internal_traffic"] = hist["is_internal_traffic"].astype(int)

    graph_artifacts = build_graph(hist)

    state.update({
        "ready": True,
        "feat": feat,
        "ae": ae,
        "gnn": gnn,
        "xgb": xgb_model,
        "history": hist,
        "graph_artifacts": graph_artifacts,
        "device": args.device,
        "seq_len": args.seq_len
    })

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
