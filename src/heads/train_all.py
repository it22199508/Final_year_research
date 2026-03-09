from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib

from .config import HEADSConfig
from .data import load_csv
from .features import add_derived_features, TabularFeaturizer
from .smote import apply_smote
from .temporal_ae import build_sequences, train_autoencoder, score_autoencoder
from .graph import build_graph
from .gnn import train_graphsage_linkpred, edge_anomaly_scores
from .xgb import train_xgboost

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV dataset")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    cfg = HEADSConfig(raw_csv=Path(args.data))
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Load data")
    df = load_csv(str(cfg.raw_csv))
    df = add_derived_features(df)

    # ------------ Temporal AE (sequence anomaly score) ------------
    print("[2] Train Transformer Autoencoder (temporal)")
    # Use a compact numeric feature set for AE (avoid one-hot explosion)
    ae_cols = ["src_port","dst_port","bytes_sent","bytes_received","bytes_total","hour","dow","is_web","is_internal_traffic"]
    df["is_internal_traffic"] = df["is_internal_traffic"].astype(int)

    X_seq, y_last, idx_last = build_sequences(df, ae_cols, seq_len=cfg.seq_len, group_col="src_ip")
    if X_seq.shape[0] == 0:
        raise RuntimeError("Not enough events per actor to build sequences. Reduce seq_len or add more data.")
    ae = train_autoencoder(X_seq, cfg, device=args.device)
    torch.save(ae.state_dict(), cfg.model_dir / "transformer_ae.pt")

    temporal_scores = score_autoencoder(ae, X_seq, device=args.device)
    # align scores to event rows (score for last event of each window)
    df["temporal_score"] = 0.0
    df.loc[idx_last, "temporal_score"] = temporal_scores

    # ------------ GraphSAGE (relational anomaly score) ------------
    print("[3] Train GraphSAGE (relational)")
    g_art = build_graph(df)
    gnn = train_graphsage_linkpred(g_art.data, cfg, device=args.device)
    torch.save(gnn.state_dict(), cfg.model_dir / "graphsage.pt")

    # score actor->resource edges per row
    import torch as _torch
    edges = []
    for _, r in df.iterrows():
        a = g_art.node_id_map[("actor", str(r["src_ip"]))]
        b = g_art.node_id_map[("resource", str(r["dst_ip"]))]
        edges.append((a,b))
    edge_index = _torch.tensor(np.array(edges, dtype=np.int64).T, dtype=_torch.long)
    rel_scores = edge_anomaly_scores(gnn, g_art.data, edge_index, device=args.device)
    df["relational_score"] = rel_scores

    # ------------ XGBoost (final classification) ------------
    print("[4] Train XGBoost (fusion)")
    if "label" not in df.columns:
        print("⚠️ No 'label' column found. Skipping supervised XGBoost training.")
        df.to_csv(cfg.processed_dir / "scored_events.csv", index=False)
        print(f"Saved scores -> {cfg.processed_dir / 'scored_events.csv'}")
        return

    feat = TabularFeaturizer(use_text=True, text_dim=64).fit(df)
    X_tab = feat.transform(df)
    X = np.hstack([X_tab, df[["temporal_score","relational_score"]].to_numpy(dtype=float)])
    y = df["label"].to_numpy(dtype=int)

    # SMOTE balancing
    X_res, y_res = apply_smote(X, y, random_state=42)

    xgb = train_xgboost(X_res, y_res, cfg)
    xgb.save_model(str(cfg.model_dir / "xgb.json"))
    joblib.dump(feat, cfg.model_dir / "tabular_featurizer.joblib")

    # Save final scored dataset
    proba = xgb.predict_proba(X)[:,1]
    df["xgb_proba"] = proba
    df["prediction"] = (proba >= 0.5).astype(int)

    df.to_csv(cfg.processed_dir / "scored_events.csv", index=False)
    print(f"Saved -> {cfg.processed_dir / 'scored_events.csv'}")
    print("Done.")

if __name__ == "__main__":
    main()
