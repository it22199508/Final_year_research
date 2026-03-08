# src/train_model.py

import os
import re
import hashlib
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from config import (
    DATA_DIR,
    MODELS_DIR,
    TRAIN_FEATURES_FILE,
    RF_MODEL_FILE,
    LOF_MODEL_FILE,
    SCALER_FILE,
    TRAIN_PRED_FILE,
)

warnings.filterwarnings("ignore")


def sha256_hex(value: object) -> str:
    s = str(value).encode("utf-8", errors="ignore")
    return hashlib.sha256(s).hexdigest()


def is_sha256_hex(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return bool(re.fullmatch(r"[0-9a-f]{64}", s.strip().lower()))


def features_user_ids_look_hashed(df: pd.DataFrame) -> bool:
    if "user_id" not in df.columns or len(df) == 0:
        return False
    sample = df["user_id"].dropna().astype(str).head(50).tolist()
    if not sample:
        return False
    return sum(is_sha256_hex(x) for x in sample) >= max(1, int(0.7 * len(sample)))


def find_ground_truth_file(data_dir: str):
    candidates = [
        "answers.csv",
        "insiders.csv",
        "insider.csv",
        "ground_truth.csv",
        "truth.csv",
        "labels.csv",
        "malicious_users.csv",
    ]

    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p

    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower() in [c.lower() for c in candidates]:
                return os.path.join(root, f)

    return None


def detect_user_column(df: pd.DataFrame):
    possible_user_cols = [
        "user", "user_id", "userid", "employee", "subject",
        "insider", "username", "person", "uid", "user_name"
    ]
    for c in possible_user_cols:
        if c in df.columns:
            return c
    return None


def normalize_ground_truth_users(gt_users, hashed=False):
    if hashed:
        return set(sha256_hex(u) for u in gt_users)
    return set(str(u) for u in gt_users)


def main():
    print("=" * 80)
    print("🚀 TRAINING MODEL ON REAL CERT r4.2 TRAIN USERS")
    print("=" * 80)

    if not os.path.exists(TRAIN_FEATURES_FILE):
        raise SystemExit(f"❌ Missing training features file: {TRAIN_FEATURES_FILE}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"📥 Loading training features: {TRAIN_FEATURES_FILE}")
    df = pd.read_csv(TRAIN_FEATURES_FILE)

    if "user_id" not in df.columns:
        raise SystemExit("❌ user_id column missing in training features.")

    print(f"✅ Loaded {len(df)} training users")
    user_ids_hashed = features_user_ids_look_hashed(df)
    print(f"🔐 user_id hashing detected: {user_ids_hashed}")

    # ---------------- LABEL CREATION ----------------
    print("\n🎯 Creating labels...")

    df["is_threat"] = 0
    gt_path = find_ground_truth_file(DATA_DIR)
    used_real_labels = False

    if gt_path and os.path.exists(gt_path):
        print(f"✅ Found ground truth file: {gt_path}")
        gt = pd.read_csv(gt_path)

        user_col = detect_user_column(gt)
        if user_col is not None:
            gt_users_raw = gt[user_col].dropna().astype(str).unique().tolist()
            gt_users_norm = normalize_ground_truth_users(gt_users_raw, user_ids_hashed)

            mask = df["user_id"].astype(str).isin(gt_users_norm)
            df.loc[mask, "is_threat"] = 1

            labeled_count = int(df["is_threat"].sum())
            print(f"✅ Real labels assigned: {labeled_count}")

            if labeled_count > 0:
                used_real_labels = True
        else:
            print("⚠️ Could not detect user column in ground truth file.")

    if not used_real_labels:
        print("⚠️ No usable ground truth found. Falling back to anomaly-based synthetic labels.")
        np.random.seed(42)

        threat_weights = {
            "after_hours_ratio": 0.25,
            "external_email_ratio": 0.20,
            "exfiltration_score": 0.30,
            "suspicious_activity": 0.15,
            "temporal_anomaly": 0.10,
        }

        df["threat_score_raw"] = 0.0

        for feature, weight in threat_weights.items():
            if feature in df.columns:
                feat_min = df[feature].min()
                feat_max = df[feature].max()
                if feat_max > feat_min:
                    normalized = (df[feature] - feat_min) / (feat_max - feat_min)
                    df["threat_score_raw"] += normalized * weight

        df["threat_score_raw"] += np.random.normal(0, 0.05, len(df))
        df["threat_score_raw"] = np.clip(df["threat_score_raw"], 0, 1)

        n_threats = max(5, int(len(df) * 0.05))
        threat_indices = df.nlargest(n_threats, "threat_score_raw").index
        df.loc[threat_indices, "is_threat"] = 1

        print(f"📌 Synthetic threat labels created: {int(df['is_threat'].sum())}")

    print(f"✅ Final threat labels: {int(df['is_threat'].sum())}")

    if len(df["is_threat"].unique()) < 2:
        raise SystemExit("❌ Only one class exists in labels. Cannot train classifier.")

    # ---------------- FEATURE SELECTION ----------------
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

    print("\n🧩 Features used for training:")
    for f in available_features:
        print(f"   - {f}")

    if len(available_features) < 5:
        raise SystemExit("❌ Not enough valid features found for model training.")

    X = df[available_features].fillna(0).values
    y = df["is_threat"].values

    # ---------------- TRAIN / VALIDATION SPLIT ----------------
    print("\n✂️ Splitting train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ---------------- RANDOM FOREST ----------------
    print("\n🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)

    y_val_pred = rf_model.predict(X_val_scaled)
    y_val_proba = rf_model.predict_proba(X_val_scaled)[:, 1]

    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred, zero_division=0)
    rec = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_val_proba)

    print("✅ Random Forest trained")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    # ---------------- LOF ----------------
    print("\n🔍 Training Local Outlier Factor...")
    lof_model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.05,
        novelty=True,
        n_jobs=-1
    )
    lof_model.fit(X_train_scaled)
    print("✅ LOF trained")

    # ---------------- SAVE MODELS ----------------
    print("\n💾 Saving models...")
    joblib.dump(rf_model, RF_MODEL_FILE)
    joblib.dump(lof_model, LOF_MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    # ---------------- TRAINING-SIDE PREDICTIONS ----------------
    X_all_scaled = scaler.transform(X)
    rf_proba_all = rf_model.predict_proba(X_all_scaled)[:, 1]
    lof_decision = lof_model.decision_function(X_all_scaled)
    lof_score = 1 / (1 + np.exp(lof_decision))

    df["rf_threat_probability"] = rf_proba_all
    df["lof_anomaly_score"] = lof_score
    df["threat_score"] = 0.6 * df["rf_threat_probability"] + 0.4 * df["lof_anomaly_score"]

    df["risk_level"] = pd.cut(
        df["threat_score"],
        bins=[0, 0.3, 0.5, 0.7, 0.9, 1],
        labels=["VERY LOW", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
        include_lowest=True
    )

    df["predicted_threat"] = df["threat_score"] > 0.7
    df = df.sort_values("threat_score", ascending=False)
    df.to_csv(TRAIN_PRED_FILE, index=False)

    # ---------------- FEATURE IMPORTANCE ----------------
    feature_importance = pd.DataFrame({
        "feature": available_features,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)

    feature_importance.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)

    # ---------------- METRICS FILE ----------------
    with open(os.path.join(MODELS_DIR, "train_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("CERT r4.2 TRAINING METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training users loaded : {len(df)}\n")
        f.write(f"Used real labels      : {used_real_labels}\n")
        f.write(f"Threat count          : {int(df['is_threat'].sum())}\n")
        f.write(f"Accuracy              : {acc:.6f}\n")
        f.write(f"Precision             : {prec:.6f}\n")
        f.write(f"Recall                : {rec:.6f}\n")
        f.write(f"F1-score              : {f1:.6f}\n")
        f.write(f"ROC AUC               : {auc:.6f}\n")
        f.write("\nClassification Report\n")
        f.write(classification_report(y_val, y_val_pred, zero_division=0))
        f.write("\nConfusion Matrix\n")
        f.write(str(confusion_matrix(y_val, y_val_pred)))
        f.write("\n")

    print("\n" + "=" * 80)
    print("✅ MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"RF model saved      : {RF_MODEL_FILE}")
    print(f"LOF model saved     : {LOF_MODEL_FILE}")
    print(f"Scaler saved        : {SCALER_FILE}")
    print(f"Train predictions   : {TRAIN_PRED_FILE}")
    print(f"Feature importance  : {os.path.join(MODELS_DIR, 'feature_importance.csv')}")
    print(f"Training metrics    : {os.path.join(MODELS_DIR, 'train_metrics.txt')}")


if __name__ == "__main__":
    main()