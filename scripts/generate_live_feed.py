"""Generate dummy live feed data to simulate streaming detections.

Usage:
  python scripts/generate_live_feed.py --out data/live/live_events.csv --rows 200 --interval 5

- Writes/overwrites the output CSV every interval seconds with new random events.
- Generates events across the 3 threat categories plus normal.
"""
from __future__ import annotations
import argparse, time, random
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def rand_ip(private=True):
    if private:
        return f"192.168.{random.randint(0,10)}.{random.randint(1,254)}"
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def make_batch(n: int):
    now = datetime.utcnow()
    rows=[]
    cats = ["Suspicious Login","Privilege Abuse","Account Takeover","Normal"]
    weights = [0.15, 0.10, 0.05, 0.70]
    for i in range(n):
        cat = random.choices(cats, weights=weights, k=1)[0]
        ts = now - timedelta(seconds=random.randint(0, 600))
        src = rand_ip(True)
        dst = rand_ip(False) if cat!="Privilege Abuse" else rand_ip(True)
        dst_port = random.choice([22,80,443,3389,445,53,8080])
        src_port = random.randint(1024,65535)
        protocol = random.choice(["TCP","UDP"])
        bytes_sent = random.randint(50, 2000)
        bytes_received = random.randint(50, 8000)
        temporal = np.clip(np.random.normal(0.02, 0.01), 0, 1)
        relational = np.clip(np.random.normal(0.02, 0.01), 0, 1)
        proba = np.clip(np.random.normal(0.10, 0.08), 0, 1)

        if cat == "Suspicious Login":
            temporal = np.clip(np.random.normal(0.20, 0.08), 0, 1)
            proba = np.clip(np.random.normal(0.55, 0.15), 0, 1)
        elif cat == "Privilege Abuse":
            relational = np.clip(np.random.normal(0.25, 0.10), 0, 1)
            proba = np.clip(np.random.normal(0.60, 0.18), 0, 1)
        elif cat == "Account Takeover":
            temporal = np.clip(np.random.normal(0.30, 0.12), 0, 1)
            relational = np.clip(np.random.normal(0.30, 0.12), 0, 1)
            proba = np.clip(np.random.normal(0.85, 0.10), 0, 1)

        rows.append({
            "timestamp": ts.isoformat(sep=" "),
            "src_ip": src,
            "dst_ip": dst,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "bytes_total": bytes_sent + bytes_received,
            "hour": ts.hour,
            "dow": ts.weekday(),
            "is_web": 1 if dst_port in (80,443,8080) else 0,
            "is_internal_traffic": 1 if dst.startswith("192.168.") else 0,
            "temporal_score": float(temporal),
            "relational_score": float(relational),
            "xgb_proba": float(proba),
            "threat_category": cat
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/live/live_events.csv")
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--interval", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing dummy live feed to {out} every {args.interval}s ... Ctrl+C to stop")
    try:
        while True:
            df = make_batch(args.rows)
            df.to_csv(out, index=False)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    main()
