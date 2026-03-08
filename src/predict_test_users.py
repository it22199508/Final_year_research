# src/predict_test_users.py

import os
import joblib
import pandas as pd
import numpy as np

from config import (
    TEST_FEATURES_FILE,
    RF_MODEL_FILE,
    LOF_MODEL_FILE,
    SCALER_FILE,
    TEST_PRED_FILE,
    TEST_DASHBOARD_FILE,
)

def main():
    print("=" * 80)
    print("🔮 PREDICTING ON UNSEEN REAL TEST USERS (CERT r4.2)")
    print("=" * 80)

    if not os.path.exists(TEST_FEATURES_FILE):
        raise SystemExit(f"❌ Missing test features file: {TEST_FEATURES_FILE}")

    if not os.path.exists(RF_MODEL_FILE):
        raise SystemExit(f"❌ Missing RF model: {RF_MODEL_FILE}")

    if not os.path.exists(LOF_MODEL_FILE):
        raise SystemExit(f"❌ Missing LOF model: {LOF_MODEL_FILE}")

    if not os.path.exists(SCALER_FILE):
        raise SystemExit(f"❌ Missing scaler: {SCALER_FILE}")

    print(f"📥 Loading test features: {TEST_FEATURES_FILE}")
    df = pd.read_csv(TEST_FEATURES_FILE)
    print(f"✅ Loaded {len(df)} unseen test users")

    print("📥 Loading trained models...")
    rf_model = joblib.load(RF_MODEL_FILE)
    lof_model = joblib.load(LOF_MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    selected_features = [
        "after_hours_ratio",
        "foreign_pc_ratio",
        "usb_connects",
        "device_intensity",
        "upload_ratio",
        "external_email_ratio",
        "copy_to_removable",
        "psych_risk",
        "activity_volume",
        "exfiltration_score",
        "suspicious_activity",
        "temporal_anomaly",
    ]

    available_features = [f for f in selected_features if f in df.columns]

    print("🧩 Features used for prediction:")
    for f in available_features:
        print(f"   - {f}")

    if len(available_features) < 5:
        raise SystemExit("❌ Not enough valid features found in test features file.")

    X = df[available_features].fillna(0).values
    X_scaled = scaler.transform(X)

    print("🤖 Generating predictions on unseen users...")
    rf_proba = rf_model.predict_proba(X_scaled)[:, 1]
    lof_decision = lof_model.decision_function(X_scaled)
    lof_score = 1 / (1 + np.exp(lof_decision))

    df["rf_threat_probability"] = rf_proba
    df["lof_anomaly_score"] = lof_score

    # Hybrid score
    df["threat_score"] = (
        0.6 * df["rf_threat_probability"] +
        0.4 * df["lof_anomaly_score"]
    )

    df["risk_level"] = pd.cut(
        df["threat_score"],
        bins=[0, 0.3, 0.5, 0.7, 0.9, 1],
        labels=["VERY LOW", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
        include_lowest=True
    )

    df["predicted_threat"] = df["threat_score"] > 0.7
    df = df.sort_values("threat_score", ascending=False)

    print("\n📊 Unseen test user summary:")
    print(f"Total unseen test users: {len(df)}")
    print(f"Predicted threats (>0.7): {int(df['predicted_threat'].sum())}")
    print(f"Critical users (>0.9): {int((df['threat_score'] > 0.9).sum())}")
    print(f"Average threat score: {df['threat_score'].mean():.4f}")

    print("\nTop 10 unseen suspicious users:")
    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
        print(
            f"{i}. {row['user_id']} | "
            f"score={row['threat_score']:.3f} | "
            f"risk={row['risk_level']} | "
            f"RF={row['rf_threat_probability']:.3f} | "
            f"LOF={row['lof_anomaly_score']:.3f}"
        )

    # Save full prediction file
    df.to_csv(TEST_PRED_FILE, index=False)

    # Save dashboard-friendly subset
    dashboard_cols = [
        "user_id",
        "threat_score",
        "risk_level",
        "predicted_threat",
        "after_hours_logons",
        "foreign_pc_logons",
        "foreign_pc_ratio",
        "usb_connects",
        "http_uploads",
        "external_emails",
        "copy_to_removable",
        "rf_threat_probability",
        "lof_anomaly_score",
        "exfiltration_score",
        "suspicious_activity",
    ]
    dashboard_cols = [c for c in dashboard_cols if c in df.columns]
    df[dashboard_cols].to_csv(TEST_DASHBOARD_FILE, index=False)

    print("\n" + "=" * 80)
    print("✅ TEST USER PREDICTION COMPLETE")
    print("=" * 80)
    print(f"Saved full test predictions : {TEST_PRED_FILE}")
    print(f"Saved dashboard test file   : {TEST_DASHBOARD_FILE}")


if __name__ == "__main__":
    main()