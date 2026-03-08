"""Baseline models for research comparison.

This project already includes sequential deep models (GRU / Transformer). This
module adds classic baselines for comparison using scikit-learn.

All baselines operate on fixed-length sequences shaped:
    X: (num_samples, seq_len, num_features)

We convert sequences into tabular vectors using a simple flattening step.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten (N, T, F) sequences into tabular (N, T*F) vectors.

    Args:
        X: NumPy array shaped (num_samples, seq_len, num_features).

    Returns:
        NumPy array shaped (num_samples, seq_len * num_features).

    Raises:
        ValueError: if X does not have exactly 3 dimensions.
    """

    X_arr = np.asarray(X)
    if X_arr.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F); got shape={X_arr.shape}")
    n, t, f = X_arr.shape
    return X_arr.reshape(int(n), int(t) * int(f))


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a Logistic Regression baseline classifier.

    Notes:
        - Flattens sequence inputs.
        - Uses class_weight='balanced' for typical class imbalance.

    Args:
        X_train: (N, T, F) training sequences.
        y_train: (N,) binary labels.

    Returns:
        Fitted sklearn LogisticRegression model.
    """

    X_flat = flatten_sequences(X_train)
    y = np.asarray(y_train).reshape(-1)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
    )
    model.fit(X_flat, y)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest baseline classifier.

    Args:
        X_train: (N, T, F) training sequences.
        y_train: (N,) binary labels.

    Returns:
        Fitted sklearn RandomForestClassifier model.
    """

    X_flat = flatten_sequences(X_train)
    y = np.asarray(y_train).reshape(-1)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_flat, y)
    return model


def train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    """Train an Isolation Forest baseline (unsupervised anomaly detection).

    This baseline does not use labels; it learns what "normal" looks like.

    Args:
        X_train: (N, T, F) training sequences.

    Returns:
        Fitted sklearn IsolationForest model.
    """

    X_flat = flatten_sequences(X_train)

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_flat)
    return model


def predict_supervised_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict P(y=1) for supervised models (logistic regression, random forest).

    Args:
        model: A fitted sklearn classifier exposing predict_proba.
        X: (N, T, F) sequences.

    Returns:
        Array of shape (N,) with probability of the positive class.

    Raises:
        ValueError: if the model does not expose predict_proba with 2 columns.
    """

    X_flat = flatten_sequences(X)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba")

    proba = np.asarray(model.predict_proba(X_flat))
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"Unexpected predict_proba shape: {proba.shape}")
    return proba[:, 1].astype(float)


def predict_isolation_forest_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Convert IsolationForest outputs into a 0..1 suspiciousness score.

    IsolationForest provides "normality" via decision_function/score_samples
    (higher = more normal). For dashboards/research comparison, it is often
    convenient to map this to a suspiciousness score where higher = more
    suspicious.

    Simple mapping used here:
        1) compute normality scores s
        2) invert them: a = -s (higher = more anomalous)
        3) min-max scale within the provided batch to [0, 1]

    Notes:
        - This is *batch-relative* scaling (depends on the evaluated X).
        - If all scores are identical, returns zeros.

    Args:
        model: Fitted IsolationForest.
        X: (N, T, F) sequences.

    Returns:
        suspiciousness: (N,) float scores in [0, 1].
    """

    X_flat = flatten_sequences(X)

    if hasattr(model, "decision_function"):
        normality = np.asarray(model.decision_function(X_flat), dtype=float).reshape(-1)
    elif hasattr(model, "score_samples"):
        normality = np.asarray(model.score_samples(X_flat), dtype=float).reshape(-1)
    else:
        raise ValueError("IsolationForest model must support decision_function or score_samples")

    anomalous = -normality
    lo = float(np.min(anomalous)) if anomalous.size else 0.0
    hi = float(np.max(anomalous)) if anomalous.size else 0.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(anomalous, dtype=float)

    return np.clip((anomalous - lo) / (hi - lo), 0.0, 1.0)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Evaluate probability-style predictions using standard binary metrics.

    Args:
        y_true: True labels (N,).
        y_prob: Predicted probabilities/scores in [0, 1] (N,).
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        Dict containing accuracy, precision, recall, f1, roc_auc,
        confusion_matrix, classification_report.
    """

    y_t = np.asarray(y_true).reshape(-1).astype(int)
    y_p = np.asarray(y_prob).reshape(-1).astype(float)

    notes: list[str] = []
    metric_notes: dict[str, str] = {}

    if y_t.size != y_p.size:
        raise ValueError(f"y_true and y_prob must have same length; got {y_t.size} vs {y_p.size}")

    # Sanitize predicted scores: ensure finite and within [0, 1] when possible.
    if y_p.size:
        nonfinite = ~np.isfinite(y_p)
        if bool(np.any(nonfinite)):
            notes.append(f"replaced {int(np.sum(nonfinite))} non-finite y_prob values with 0.0")
            y_p = y_p.copy()
            y_p[nonfinite] = 0.0

        min_p = float(np.min(y_p))
        max_p = float(np.max(y_p))
        if min_p < 0.0 or max_p > 1.0:
            notes.append(f"clipped y_prob to [0,1] (range was {min_p:.4f}..{max_p:.4f})")
            y_p = np.clip(y_p, 0.0, 1.0)

    y_pred = (y_p >= float(threshold)).astype(int)

    out: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_t, y_pred)) if y_t.size else 0.0,
        "precision": float(precision_score(y_t, y_pred, zero_division=0)) if y_t.size else 0.0,
        "recall": float(recall_score(y_t, y_pred, zero_division=0)) if y_t.size else 0.0,
        "f1": float(f1_score(y_t, y_pred, zero_division=0)) if y_t.size else 0.0,
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_t, y_pred, labels=[0, 1]).tolist()
        if y_t.size
        else [[0, 0], [0, 0]],
        "classification_report": classification_report(y_t, y_pred, zero_division=0) if y_t.size else "",
        "notes": "; ".join(notes) if notes else "",
        "metric_notes": metric_notes,
    }

    # Safe ROC-AUC: requires both classes present.
    # Safe ROC-AUC: requires both classes present.
    if y_t.size == 0:
        metric_notes["roc_auc"] = "roc_auc unavailable: empty y_true"
        out["roc_auc"] = None
    elif np.unique(y_t).size < 2:
        metric_notes["roc_auc"] = "roc_auc unavailable: only one class present in y_true"
        out["roc_auc"] = None
    else:
        try:
            out["roc_auc"] = float(roc_auc_score(y_t, y_p))
        except Exception as e:
            metric_notes["roc_auc"] = f"roc_auc unavailable: {type(e).__name__}: {e}"
            out["roc_auc"] = None

    return out


if __name__ == "__main__":
    # Small demo using synthetic sequence data.
    rng = np.random.default_rng(0)

    X = rng.normal(size=(200, 10, 6)).astype(np.float32)
    y = (rng.random(200) > 0.85).astype(int)  # imbalanced positives

    # Inject a simple anomaly signal into positives.
    X[y == 1] += 1.0

    lr = train_logistic_regression(X, y)
    rf = train_random_forest(X, y)
    iso = train_isolation_forest(X)

    lr_prob = predict_supervised_probabilities(lr, X)
    rf_prob = predict_supervised_probabilities(rf, X)
    iso_prob = predict_isolation_forest_scores(iso, X)

    print("LogReg:", evaluate_predictions(y, lr_prob)["roc_auc"])
    print("RF:", evaluate_predictions(y, rf_prob)["roc_auc"])
    print("IsoForest:", evaluate_predictions(y, iso_prob)["roc_auc"])
