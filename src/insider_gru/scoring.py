"""Explainable manual scoring for insider-threat events.

This module implements research-friendly *single-event* scoring based on a small
set of deterministic rules.

Key properties:
- Inputs: a 1-row engineered DataFrame (from :mod:`insider_gru.features`).
- Config: :class:`insider_gru.config.ScoringConfig` dataclass.
- Outputs: a score in $[0,1]$, an explainable per-rule breakdown, and a simple
    decision label (ALERT/OK).

If a required feature is missing, it is treated as 0.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from insider_gru.config import ScoringConfig


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _row_value(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return 0.0
    v = row[col]
    try:
        if pd.isna(v):
            return 0.0
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return 0.0


def _rule_entry(
    *,
    rule: str,
    weight: float,
    rule_score: float,
    details: str,
) -> Dict[str, Any]:
    rs = _clamp01(float(rule_score))
    w = float(weight)
    return {
        "rule": str(rule),
        "triggered": bool(rs > 0.0),
        "weight": w,
        "rule_score": rs,
        "contribution": _clamp01(w * rs),
        "details": str(details),
    }


def _content_text(row: pd.Series) -> str:
    # The CERT-style dataset uses 'content'. Dashboards sometimes use other names.
    for col in ("content", "body", "email_content", "text"):
        if col in row.index:
            try:
                v = row[col]
                if pd.isna(v):
                    continue
                return str(v)
            except Exception:
                continue
    return ""


def _confidence_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def score_email_event(features_df: pd.DataFrame, cfg: ScoringConfig) -> Dict[str, Any]:
    """Compute a manual score for a single engineered email event.

    Args:
        features_df: DataFrame with exactly one row containing engineered features.
        cfg: ScoringConfig with thresholds and weights.

    Returns:
        Dict with:
            - manual_score: float
            - triggered_rules: list[str]
            - confidence_label: str
            - threshold: float
            - decision: str (ALERT/OK)
            - breakdown: list[dict[str, Any]] (per-rule contributions)

    Raises:
        ValueError: If features_df does not contain exactly one row.
    """

    if features_df.shape[0] != 1:
        raise ValueError("score_email_event expects a single-row DataFrame")

    row = features_df.iloc[0]

    triggered: List[str] = []
    breakdown: List[Dict[str, Any]] = []

    base = _clamp01(float(cfg.base_risk_score))
    if base > 0.0:
        breakdown.append(
            _rule_entry(
                rule="base_risk_score",
                weight=1.0,
                rule_score=base,
                details=f"base_risk_score={base:.2f}",
            )
        )
        triggered.append("base_risk_score")

    # Rule: after-hours email
    ah = 1.0 if _row_value(row, "is_after_hours") >= 1.0 else 0.0
    breakdown.append(
        _rule_entry(
            rule="after_hours",
            weight=float(cfg.after_hours_weight),
            rule_score=ah,
            details=f"is_after_hours={int(ah)}",
        )
    )
    if ah > 0:
        triggered.append("after_hours")

    # Rule: large content size (linear ramp above threshold)
    threshold_bytes = max(0, _safe_int(cfg.content_size_threshold, 0))
    size = max(0.0, _row_value(row, "content_size"))
    if threshold_bytes <= 0:
        large_score = 0.0
        large_details = f"content_size_threshold={threshold_bytes} (disabled)"
    else:
        large_score = _clamp01((size - threshold_bytes) / float(threshold_bytes)) if size > threshold_bytes else 0.0
        large_details = f"content_size={int(size)}, threshold={threshold_bytes}"
    breakdown.append(
        _rule_entry(
            rule="large_content",
            weight=float(cfg.large_content_weight),
            rule_score=large_score,
            details=large_details,
        )
    )
    if large_score > 0:
        triggered.append("large_content")

    # Rule: attachments (linear ramp above threshold)
    att_threshold = max(0, _safe_int(cfg.attachment_threshold, 0))
    att_count = max(0.0, _row_value(row, "attachment_count"))
    if att_threshold <= 0:
        att_score = 0.0
        att_details = f"attachment_threshold={att_threshold} (disabled)"
    else:
        denom = float(max(1, att_threshold))
        att_score = _clamp01((att_count - att_threshold) / denom) if att_count > att_threshold else 0.0
        att_details = f"attachment_count={int(att_count)}, threshold={att_threshold}"
    breakdown.append(
        _rule_entry(
            rule="attachments",
            weight=float(cfg.attachment_weight),
            rule_score=att_score,
            details=att_details,
        )
    )
    if att_score > 0:
        triggered.append("attachments")

    # Rule: external recipient
    ext = 1.0 if _row_value(row, "external_recipient_flag") >= 1.0 else 0.0
    breakdown.append(
        _rule_entry(
            rule="external_recipient",
            weight=float(cfg.external_recipient_weight),
            rule_score=ext,
            details=f"external_recipient_flag={int(ext)}",
        )
    )
    if ext > 0:
        triggered.append("external_recipient")

    # Rule: suspicious keywords (fraction of configured keywords matched)
    keywords = [str(k).strip().lower() for k in (cfg.suspicious_keywords or []) if str(k).strip()]
    body = _content_text(row).lower()
    matched: list[str] = []
    if keywords and body:
        for kw in keywords:
            if kw and kw in body:
                matched.append(kw)
    kw_score = (len(set(matched)) / float(len(keywords))) if keywords else 0.0
    kw_details = "no keywords" if not keywords else f"matched={sorted(set(matched))}"
    breakdown.append(
        _rule_entry(
            rule="suspicious_keywords",
            weight=float(cfg.suspicious_keyword_weight),
            rule_score=kw_score,
            details=kw_details,
        )
    )
    if kw_score > 0:
        triggered.append("suspicious_keywords")

    manual_score = _clamp01(sum(float(r.get("contribution", 0.0)) for r in breakdown))
    threshold = float(cfg.threshold)
    decision = "ALERT" if manual_score >= threshold else "OK"

    return {
        "manual_score": manual_score,
        "triggered_rules": triggered,
        "confidence_label": _confidence_from_score(manual_score),
        "threshold": threshold,
        "decision": decision,
        "breakdown": breakdown,
    }


def score_login_event(features_df: pd.DataFrame, cfg: ScoringConfig) -> Dict[str, Any]:
    """Compute a manual score for a single engineered login/device event."""

    if features_df.shape[0] != 1:
        raise ValueError("score_login_event expects a single-row DataFrame")

    row = features_df.iloc[0]

    triggered: List[str] = []
    breakdown: List[Dict[str, Any]] = []

    base = _clamp01(float(cfg.base_risk_score))
    if base > 0.0:
        breakdown.append(
            _rule_entry(
                rule="base_risk_score",
                weight=1.0,
                rule_score=base,
                details=f"base_risk_score={base:.2f}",
            )
        )
        triggered.append("base_risk_score")

    # Rule: after-hours login
    ah = 1.0 if _row_value(row, "after_hours_login") >= 1.0 else 0.0
    breakdown.append(
        _rule_entry(
            rule="after_hours_login",
            weight=float(cfg.after_hours_weight),
            rule_score=ah,
            details=f"after_hours_login={int(ah)}",
        )
    )
    if ah > 0:
        triggered.append("after_hours_login")

    # Rule: weekend login
    wk = 1.0 if _row_value(row, "is_weekend") >= 1.0 else 0.0
    breakdown.append(
        _rule_entry(
            rule="weekend_login",
            weight=float(cfg.weekend_login_weight),
            rule_score=wk,
            details=f"is_weekend={int(wk)}",
        )
    )
    if wk > 0:
        triggered.append("weekend_login")

    # Rule: high recent login frequency (linear ramp above threshold)
    lf_threshold = max(0, _safe_int(cfg.login_frequency_threshold, 0))
    lf = max(0.0, _row_value(row, "login_frequency_recent_window"))
    if lf_threshold <= 0:
        lf_score = 0.0
        lf_details = f"login_frequency_threshold={lf_threshold} (disabled)"
    else:
        denom = float(max(1, lf_threshold))
        lf_score = _clamp01((lf - lf_threshold) / denom) if lf > lf_threshold else 0.0
        lf_details = f"login_frequency_recent_window={int(lf)}, threshold={lf_threshold}"
    breakdown.append(
        _rule_entry(
            rule="high_login_frequency",
            weight=float(cfg.high_login_frequency_weight),
            rule_score=lf_score,
            details=lf_details,
        )
    )
    if lf_score > 0:
        triggered.append("high_login_frequency")

    # Rule: unusual device
    ud = 1.0 if _row_value(row, "unusual_device_flag") >= 1.0 else 0.0
    breakdown.append(
        _rule_entry(
            rule="unusual_device",
            weight=float(cfg.unusual_device_weight),
            rule_score=ud,
            details=f"unusual_device_flag={int(ud)}",
        )
    )
    if ud > 0:
        triggered.append("unusual_device")

    manual_score = _clamp01(sum(float(r.get("contribution", 0.0)) for r in breakdown))
    threshold = float(cfg.threshold)
    decision = "ALERT" if manual_score >= threshold else "OK"

    return {
        "manual_score": manual_score,
        "triggered_rules": triggered,
        "confidence_label": _confidence_from_score(manual_score),
        "threshold": threshold,
        "decision": decision,
        "breakdown": breakdown,
    }
