"""Core orchestration pipeline for insider threat detection.

This module connects the project's building blocks:
- preprocessing (cleaning/normalization/hashing)
- feature engineering (behavioral features)
- sequence building (fixed-length windows for GRU/Transformer)
- scoring (manual / hybrid combination)

It is intentionally simple and explainable, suitable for research dashboards,
JSON export, and future live event simulation.

Notes
-----
- This pipeline processes a *single* raw event at a time.
- Recent-window behavioral features in `features` require history; for single
  events they will naturally be 0 unless you extend this pipeline to inject
  per-user history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4

import pandas as pd

from insider_gru import features, preprocessing, scoring, sequences  # sequences is imported for pipeline completeness
from insider_gru.config import ScoringConfig


@dataclass(frozen=True)
class PipelineConfig:
    """Typed view over the free-form config dict."""

    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    recent_window_hours: int = 24

    # Column names (defaults reflect current repo conventions)
    email_timestamp_col: str = "shared_time"
    login_timestamp_col: str = "login_time"

    # Optional PII hashing controls
    hash_pii: bool = False
    pii_columns: Optional[list[str]] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_col_name(name: str) -> str:
    """Match preprocessing/features column normalization (lowercase underscores)."""

    s = str(name).strip().lower()
    for ch in (" ", "-", ".", "/", "\\", ":", "\t", "\n", "\r"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _jsonable(value: Any) -> Any:
    """Convert common pandas/numpy scalars to JSON-friendly python types."""

    # pandas Timestamp
    if hasattr(value, "isoformat") and isinstance(value, (pd.Timestamp, datetime)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)

    # numpy scalars
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
    except Exception:
        pass

    # missing values
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def _df_row_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    if df.shape[0] != 1:
        raise ValueError("Expected a single-row DataFrame")

    row = df.iloc[0].to_dict()
    return {str(k): _jsonable(v) for k, v in row.items()}


def _canonicalize_email_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical email identity columns exist: sender, recipient.

    The preprocessing step normalizes column *names* but does not rename semantic
    aliases (e.g., from/to). For consistent downstream processing and reporting,
    this helper adds canonical columns when aliases exist.
    """

    out = df.copy(deep=True)

    if "sender" not in out.columns and "from" in out.columns:
        out["sender"] = out["from"]
    if "recipient" not in out.columns and "to" in out.columns:
        out["recipient"] = out["to"]

    return out


def _canonicalize_login_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical login identity columns exist: user."""

    out = df.copy(deep=True)

    if "user" not in out.columns and "username" in out.columns:
        out["user"] = out["username"]

    return out


def _parse_config(config: Dict[str, Any]) -> PipelineConfig:
    """Parse a free-form dict into a typed PipelineConfig with safe defaults."""

    def _get_int(key: str, default: int) -> int:
        try:
            return int(config.get(key, default))
        except Exception:
            return default

    def _get_float(key: str, default: float) -> float:
        try:
            return float(config.get(key, default))
        except Exception:
            return default

    def _get_bool(key: str, default: bool) -> bool:
        v = config.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "y"}
        return bool(v)

    pii_cols = config.get("pii_columns")
    if pii_cols is not None and not isinstance(pii_cols, list):
        pii_cols = None

    # Build ScoringConfig from dict using the same field names used in scoring.py and dashboards.
    kw = config.get("suspicious_keywords")
    if not isinstance(kw, list):
        kw = ScoringConfig().suspicious_keywords

    scoring_cfg = ScoringConfig(
        threshold=_get_float("threshold", ScoringConfig.threshold),
        scoring_mode=str(config.get("scoring_mode", ScoringConfig.scoring_mode)),
        hybrid_alpha=_get_float("hybrid_alpha", ScoringConfig.hybrid_alpha),
        base_risk_score=_get_float("base_risk_score", ScoringConfig.base_risk_score),
        after_hours_weight=_get_float("after_hours_weight", ScoringConfig.after_hours_weight),
        large_content_weight=_get_float("large_content_weight", ScoringConfig.large_content_weight),
        attachment_weight=_get_float("attachment_weight", ScoringConfig.attachment_weight),
        external_recipient_weight=_get_float("external_recipient_weight", ScoringConfig.external_recipient_weight),
        suspicious_keyword_weight=_get_float("suspicious_keyword_weight", ScoringConfig.suspicious_keyword_weight),
        weekend_login_weight=_get_float("weekend_login_weight", ScoringConfig.weekend_login_weight),
        high_login_frequency_weight=_get_float(
            "high_login_frequency_weight", ScoringConfig.high_login_frequency_weight
        ),
        unusual_device_weight=_get_float("unusual_device_weight", ScoringConfig.unusual_device_weight),
        content_size_threshold=_get_int("content_size_threshold", ScoringConfig.content_size_threshold),
        attachment_threshold=_get_int("attachment_threshold", ScoringConfig.attachment_threshold),
        login_frequency_threshold=_get_int("login_frequency_threshold", ScoringConfig.login_frequency_threshold),
        business_hours_start=_get_int("business_hours_start", ScoringConfig.business_hours_start),
        business_hours_end=_get_int("business_hours_end", ScoringConfig.business_hours_end),
        suspicious_keywords=list(kw),
    )

    return PipelineConfig(
        scoring=scoring_cfg,
        recent_window_hours=_get_int("recent_window_hours", 24),
        email_timestamp_col=str(config.get("email_timestamp_col", "shared_time")),
        login_timestamp_col=str(config.get("login_timestamp_col", "login_time")),
        hash_pii=_get_bool("hash_pii", False),
        pii_columns=pii_cols,
    )


def _normalize_scoring_mode(mode: str) -> str:
    """Normalize human-friendly scoring mode labels.

    Supports:
    - 'model', 'manual', 'hybrid' (pipeline internal)
    - 'Model Prediction', 'Manual Scoring', 'Hybrid' (dashboards)
    """

    m = (mode or "").strip().lower()
    if m in {"model", "model prediction", "prediction", "model_prediction"}:
        return "model"
    if m in {"manual", "manual scoring", "manual_score", "manual_scoring"}:
        return "manual"
    if m in {"hybrid", "blend", "combined"}:
        return "hybrid"
    return m


def prepare_single_event_dataframe(raw_event: Dict[str, Any]) -> pd.DataFrame:
    """Convert a single raw event dict into a 1-row DataFrame.

    Args:
        raw_event: Dict containing raw event fields.

    Returns:
        A pandas DataFrame with one row.

    Raises:
        ValueError: If raw_event is not a dict.
    """

    if not isinstance(raw_event, dict):
        raise ValueError("raw_event must be a dict")

    return pd.DataFrame([raw_event])


def process_email_event(raw_event: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single email event through clean → features → manual scoring.

    Returns a structured dict suitable for alerting/reporting.
    """

    cfg = _parse_config(config)

    df = prepare_single_event_dataframe(raw_event)

    ts_col = cfg.email_timestamp_col
    ts_norm = _normalize_col_name(ts_col)

    pii_columns = cfg.pii_columns
    if pii_columns is None:
        pii_columns = ["sender", "recipient", "from", "to", "user", "username", "device", "ip"]

    cleaned = preprocessing.clean_event_data(
        df,
        timestamp_columns=[ts_col],
        pii_columns=pii_columns if cfg.hash_pii else None,
    )

    cleaned = _canonicalize_email_columns(cleaned)

    if ts_norm not in cleaned.columns:
        raise ValueError(f"Email timestamp column not found after cleaning: {ts_col!r}")

    scoring_cfg = cfg.scoring

    engineered = features.engineer_email_features(
        cleaned,
        timestamp_col=ts_norm,
        business_hours_start=int(scoring_cfg.business_hours_start),
        business_hours_end=int(scoring_cfg.business_hours_end),
        recent_window_hours=int(cfg.recent_window_hours),
    )

    manual = scoring.score_email_event(engineered, scoring_cfg)

    processed_at = _now_iso()

    return {
        "event_type": "email",
        "cleaned_inputs": _df_row_to_dict(cleaned),
        "engineered_features": _df_row_to_dict(engineered),
        "manual_score": float(manual["manual_score"]),
        "triggered_rules": list(manual.get("triggered_rules", [])),
        "confidence_label": str(manual.get("confidence_label", "low")),
        "threshold": float(scoring_cfg.threshold),
        "decision": str(manual.get("decision", "OK")),
        "processed_at": processed_at,
    }


def process_login_event(raw_event: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single login/device event through clean → features → manual scoring."""

    cfg = _parse_config(config)

    df = prepare_single_event_dataframe(raw_event)

    ts_col = cfg.login_timestamp_col
    ts_norm = _normalize_col_name(ts_col)

    pii_columns = cfg.pii_columns
    if pii_columns is None:
        pii_columns = ["user", "username", "device", "ip", "src_ip", "dst_ip"]

    cleaned = preprocessing.clean_event_data(
        df,
        timestamp_columns=[ts_col],
        pii_columns=pii_columns if cfg.hash_pii else None,
    )

    cleaned = _canonicalize_login_columns(cleaned)

    if ts_norm not in cleaned.columns:
        raise ValueError(f"Login timestamp column not found after cleaning: {ts_col!r}")

    scoring_cfg = cfg.scoring

    engineered = features.engineer_login_features(
        cleaned,
        timestamp_col=ts_norm,
        business_hours_start=int(scoring_cfg.business_hours_start),
        business_hours_end=int(scoring_cfg.business_hours_end),
        recent_window_hours=int(cfg.recent_window_hours),
    )

    manual = scoring.score_login_event(engineered, scoring_cfg)

    processed_at = _now_iso()

    return {
        "event_type": "login",
        "cleaned_inputs": _df_row_to_dict(cleaned),
        "engineered_features": _df_row_to_dict(engineered),
        "manual_score": float(manual["manual_score"]),
        "triggered_rules": list(manual.get("triggered_rules", [])),
        "confidence_label": str(manual.get("confidence_label", "low")),
        "threshold": float(scoring_cfg.threshold),
        "decision": str(manual.get("decision", "OK")),
        "processed_at": processed_at,
    }


def combine_pipeline_scores(
    model_score: float | None,
    manual_result: Dict[str, Any],
    mode: str,
    alpha: float,
    threshold: float,
) -> Dict[str, Any]:
    """Combine model and manual scores into a final score and decision.

    Modes:
        - "model": use only model_score (requires model_score).
        - "manual": use only manual_score from manual_result.
        - "hybrid": blend model and manual: final = alpha*model + (1-alpha)*manual

    Confidence label:
        A simple, explainable heuristic based on how far final_score is from the
        threshold.

    Args:
        model_score: Optional model probability/score in [0, 1].
        manual_result: Dict containing at least manual_score.
        mode: "model" | "manual" | "hybrid".
        alpha: Model weight in hybrid mode.
        threshold: Decision threshold.

    Returns:
        Dict with model_score, manual_score, final_score, confidence_label, decision.

    Raises:
        ValueError: If required inputs are missing.
    """

    manual_score = float(manual_result.get("manual_score", 0.0))

    m = _normalize_scoring_mode(mode)
    a = float(alpha)
    a = max(0.0, min(1.0, a))

    if m == "model":
        if model_score is None:
            raise ValueError("mode='model' requires model_score")
        final = float(model_score)
    elif m == "manual":
        final = manual_score
    elif m == "hybrid":
        if model_score is None:
            # Research-friendly fallback: if model is absent, behave like manual.
            final = manual_score
        else:
            final = a * float(model_score) + (1.0 - a) * manual_score
    else:
        raise ValueError(f"Unsupported scoring mode: {mode!r}. Use 'model', 'manual', or 'hybrid'.")

    final = max(0.0, min(1.0, float(final)))

    # Explainable confidence based on margin from the threshold.
    margin = abs(final - float(threshold))
    if margin >= 0.20:
        confidence = "high"
    elif margin >= 0.10:
        confidence = "medium"
    else:
        confidence = "low"

    decision = "ALERT" if final >= float(threshold) else "OK"

    return {
        "model_score": None if model_score is None else float(model_score),
        "manual_score": manual_score,
        "final_score": final,
        "confidence_label": confidence,
        "decision": decision,
    }


def run_detection_pipeline(
    raw_event: Dict[str, Any],
    event_type: str,
    config: Dict[str, Any],
    model_score: float | None = None,
) -> Dict[str, Any]:
    """Run the end-to-end pipeline for a single event.

    Steps:
        1) Route to event-specific processing (clean + features + manual scoring)
        2) Combine manual score with optional model score
        3) Return an alert-style dict for dashboards / JSON export

    Args:
        raw_event: Raw event dict.
        event_type: "email" or "login".
        config: Free-form config dict.
        model_score: Optional model probability/score.

    Returns:
        Final alert dict containing fields documented in the project spec.
    """

    cfg = _parse_config(config)

    et = (event_type or "").strip().lower()
    if et in {"email", "emails"}:
        manual_result = process_email_event(raw_event, config)
    elif et in {"login", "logins", "device", "auth", "authentication"}:
        manual_result = process_login_event(raw_event, config)
        et = "login"
    else:
        raise ValueError(f"Unsupported event_type: {event_type!r}. Use 'email' or 'login'.")

    combined = combine_pipeline_scores(
        model_score=model_score,
        manual_result=manual_result,
        mode=cfg.scoring.scoring_mode,
        alpha=cfg.scoring.hybrid_alpha,
        threshold=cfg.scoring.threshold,
    )

    alert_id = str(uuid4())

    return {
        "alert_id": alert_id,
        "event_type": et,
        "cleaned_inputs": manual_result["cleaned_inputs"],
        "engineered_features": manual_result["engineered_features"],
        "model_score": combined["model_score"],
        "manual_score": combined["manual_score"],
        "final_score": combined["final_score"],
        "triggered_rules": manual_result.get("triggered_rules", []),
        "confidence_label": combined["confidence_label"],
        "threshold": float(cfg.scoring.threshold),
        "decision": combined["decision"],
        "processed_at": manual_result.get("processed_at", _now_iso()),
    }


if __name__ == "__main__":
    # Minimal demo showing how to run the orchestration layer.
    # This can later be wired into Streamlit dashboards or a live event simulator.

    demo_config: Dict[str, Any] = {
        "business_hours_start": 8,
        "business_hours_end": 18,
        "recent_window_hours": 24,
        "scoring_mode": "Hybrid",
        "hybrid_alpha": 0.6,
        "threshold": 0.5,
        "hash_pii": False,
        # Optional output path placeholder (not used here, but common in pipelines)
        "output_dir": str(Path("outputs") / "alerts"),
    }

    sample_email_event = {
        "shared_time": "2026-03-07 23:10:00",
        "sender": "alice@company.com",
        "recipient": "bob@external.com",
        "attachment_count": 2,
        "content_size": 120_000,
        "subject": "Quarterly report",
    }

    sample_login_event = {
        "login_time": "2026-03-07 02:30:00",
        "user": "alice",
        "device": "LAPTOP-123",
        "ip": "10.0.0.5",
    }

    print("--- Email event (manual + hybrid combine) ---")
    email_alert = run_detection_pipeline(sample_email_event, "email", demo_config, model_score=0.55)
    print(email_alert)

    print("\n--- Login event (manual + hybrid combine) ---")
    login_alert = run_detection_pipeline(sample_login_event, "login", demo_config, model_score=0.40)
    print(login_alert)
