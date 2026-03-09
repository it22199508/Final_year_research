# src/evaluate_test_model.py

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from config import TEST_PRED_FILE


def create_behavior_ground_truth(df):
    """
    Create pseudo ground truth using anomaly ranking.
    Top 10% highest behavioral anomaly = threat.
    """

    df = df.copy()

    anomaly_score = (
        df["after_hours_logons"].rank(pct=True) +
        df["foreign_pc_ratio"].rank(pct=True) +
        df["copy_to_removable"].rank(pct=True) +
        df["external_emails"].rank(pct=True) +
        df["http_uploads"].rank(pct=True)
    ) / 5

    df["pseudo_label"] = 0

    threshold = anomaly_score.quantile(0.90)

    df.loc[anomaly_score >= threshold, "pseudo_label"] = 1

    return df


def main():

    print("=" * 80)
    print("📊 TEST SET EVALUATION (UNSEEN USERS)")
    print("=" * 80)

    df = pd.read_csv(TEST_PRED_FILE)

    print(f"Loaded test users: {len(df)}")

    df = create_behavior_ground_truth(df)

    y_true = df["pseudo_label"]
    y_pred = df["predicted_threat"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n📈 Evaluation Metrics (Test Users)")
    print("-----------------------------------")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(confusion_matrix(y_true, y_pred))

    print("\nDetailed Report")
    print("---------------")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("\nThreat Distribution")
    print("-------------------")
    print(df["risk_level"].value_counts())


if __name__ == "__main__":
    main()