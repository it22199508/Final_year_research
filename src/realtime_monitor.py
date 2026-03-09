# src/realtime_monitor.py

import json
import os
import time
from datetime import datetime

import pandas as pd

from config import (
    TEST_PRED_FILE,
    REALTIME_ALERTS_FILE,
    REALTIME_STATE_FILE,
    THREAT_THRESHOLD,
    CRITICAL_THRESHOLD,
    MONITOR_INTERVAL,
    TOP_N_LOOP,
)


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def append_alert(alert):
    alerts = load_json(REALTIME_ALERTS_FILE, [])
    alerts.insert(0, alert)
    alerts = alerts[:300]
    save_json(REALTIME_ALERTS_FILE, alerts)


def get_alert_level(score):
    if score >= CRITICAL_THRESHOLD:
        return "CRITICAL"
    elif score >= THREAT_THRESHOLD:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    elif score >= 0.3:
        return "LOW"
    return "VERY LOW"


def main():
    print("=" * 80)
    print("🚨 NSADM REAL-TIME MONITOR (CERT r4.2 TEST USERS)")
    print("=" * 80)

    if not os.path.exists(TEST_PRED_FILE):
        raise SystemExit(f"❌ Missing test prediction file: {TEST_PRED_FILE}")

    df = pd.read_csv(TEST_PRED_FILE)

    if "threat_score" not in df.columns:
        raise SystemExit("❌ threat_score column missing in test predictions")

    df = df.sort_values("threat_score", ascending=False).reset_index(drop=True)

    state = load_json(REALTIME_STATE_FILE, {"last_index": 0, "alert_counter": 0})
    last_index = state.get("last_index", 0)
    alert_counter = state.get("alert_counter", 0)

    print(f"Loaded unseen test predictions: {len(df)}")
    print(f"Threat threshold   : {THREAT_THRESHOLD}")
    print(f"Critical threshold : {CRITICAL_THRESHOLD}")
    print(f"Monitor interval   : {MONITOR_INTERVAL} sec")
    print("Press Ctrl + C to stop")
    print("=" * 80)

    try:
        while True:
            batch = df.iloc[last_index:last_index + TOP_N_LOOP]

            if len(batch) == 0:
                print("🔁 End of test-user stream reached. Restarting from top...")
                last_index = 0
                continue

            for _, row in batch.iterrows():
                score = float(row.get("threat_score", 0))

                if score < THREAT_THRESHOLD:
                    continue

                alert_counter += 1
                risk_level = get_alert_level(score)

                event = {
                    "alert_id": alert_counter,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user_id": str(row.get("user_id", "UNKNOWN")),
                    "assigned_pc": str(row.get("assigned_pc", "UNKNOWN")),
                    "last_seen_pc": str(row.get("last_seen_pc", "UNKNOWN")),
                    "threat_score": round(score, 4),
                    "risk_level": risk_level,
                    "rf_threat_probability": round(float(row.get("rf_threat_probability", 0)), 4),
                    "lof_anomaly_score": round(float(row.get("lof_anomaly_score", 0)), 4),
                    "after_hours_logons": int(row.get("after_hours_logons", 0)),
                    "foreign_pc_logons": int(row.get("foreign_pc_logons", 0)),
                    "foreign_pc_ratio": round(float(row.get("foreign_pc_ratio", 0)), 4),
                    "usb_connects": int(row.get("usb_connects", 0)),
                    "http_uploads": int(row.get("http_uploads", 0)),
                    "external_emails": int(row.get("external_emails", 0)),
                    "copy_to_removable": int(row.get("copy_to_removable", 0)),
                    "exfiltration_score": round(float(row.get("exfiltration_score", 0)), 4),
                    "suspicious_activity": round(float(row.get("suspicious_activity", 0)), 4),
                }

                append_alert(event)

                print(
                    f"⚠️ Alert #{event['alert_id']}: {event['user_id']} | "
                    f"PC={event['last_seen_pc']} | "
                    f"score={event['threat_score']:.3f} | "
                    f"risk={event['risk_level']}"
                )

                time.sleep(MONITOR_INTERVAL)

            last_index += TOP_N_LOOP
            save_json(REALTIME_STATE_FILE, {
                "last_index": last_index,
                "alert_counter": alert_counter
            })

    except KeyboardInterrupt:
        save_json(REALTIME_STATE_FILE, {
            "last_index": last_index,
            "alert_counter": alert_counter
        })
        print("\n🛑 Monitor stopped safely.")


if __name__ == "__main__":
    main()