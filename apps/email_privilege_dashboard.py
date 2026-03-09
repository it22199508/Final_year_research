from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from datetime import datetime
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from insider_gru.config import OutputConfig, ScoringConfig
from insider_gru.data import prepare_dataframe
from insider_gru.model import GRUClassifier
from insider_gru.pipeline import combine_pipeline_scores
from insider_gru import features as feature_utils
from insider_gru import scoring as manual_scoring


st.set_page_config(page_title="Email Privilege Misuse Detector", layout="wide")


_FIXED_THRESHOLD = 0.50


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _parse_keywords(csv_text: str) -> list[str]:
    """Parse comma-separated keywords; returns lowercase tokens with empties removed."""
    if not csv_text:
        return []
    parts = [p.strip().lower() for p in csv_text.split(",")]
    return [p for p in parts if p]


def _hour_in_window(hour: int, start_hour: int, end_hour: int) -> bool:
    """Return True if hour is within [start_hour, end_hour) allowing wrap across midnight."""
    start = int(start_hour)
    end = int(end_hour)
    h = int(hour)

    # Normal window (e.g., 9..17)
    if start < end:
        return start <= h < end
    # Wrapped window (e.g., 22..6)
    return (h >= start) or (h < end)


def _manual_score_email(
    *,
    sender: str,
    recipient: str,
    dt: pd.Timestamp,
    size_bytes: int,
    attachments: int,
    content: str,
    cfg: ScoringConfig,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute a manual risk score and per-rule breakdown.

    This dashboard uses the same single-event scoring logic as `insider_gru.scoring`,
    but constructs a 1-row feature DataFrame from the user's inputs.
    """

    hour = int(pd.Timestamp(dt).hour)
    in_business = _hour_in_window(hour, int(cfg.business_hours_start), int(cfg.business_hours_end))
    is_after_hours = 0 if in_business else 1
    external_flag = feature_utils.flag_external_recipient(str(sender), str(recipient))

    features_df = pd.DataFrame(
        [
            {
                "is_after_hours": int(is_after_hours),
                "content_size": float(max(0, int(size_bytes))),
                "attachment_count": int(max(0, int(attachments))),
                "external_recipient_flag": int(external_flag),
                "content": str(content or ""),
            }
        ]
    )

    result = manual_scoring.score_email_event(features_df, cfg)
    return float(result.get("manual_score", 0.0)), list(result.get("breakdown", []))


def _final_score_and_decision(
    *,
    model_probability: float,
    manual_score: float,
    cfg: ScoringConfig,
    threshold: float,
) -> tuple[float, bool, str]:
    """Compute final score based on mode and return (score, is_alert, decision)."""

    # Keep existing behavior for non-hybrid modes via the shared pipeline helper.
    if str(cfg.scoring_mode) != "Hybrid":
        combined = combine_pipeline_scores(
            model_score=_clamp01(float(model_probability)),
            manual_result={"manual_score": _clamp01(float(manual_score))},
            mode=str(cfg.scoring_mode),
            alpha=float(cfg.hybrid_alpha),
            threshold=float(threshold),
        )

        score = float(combined.get("final_score", 0.0))
        decision = str(combined.get("decision", "OK"))
        is_alert = decision.upper() == "ALERT"
        return score, bool(is_alert), decision

    model_weight = _clamp01(float(cfg.hybrid_alpha))
    manual_weight = _clamp01(1.0 - float(model_weight))

    manual_score_clamped = _clamp01(float(manual_score))

    model_score_value: float | None
    try:
        mp = float(model_probability)
        if np.isnan(mp):
            model_score_value = None
        else:
            model_score_value = _clamp01(mp)
    except Exception:
        model_score_value = None

    # Safe fallback: if the model score is unavailable, use the manual score only.
    if model_score_value is None:
        score = manual_score_clamped
    else:
        score = (model_weight * model_score_value) + (manual_weight * manual_score_clamped)

    decision = "ALERT" if float(score) >= float(threshold) else "OK"
    is_alert = decision == "ALERT"
    return float(score), bool(is_alert), str(decision)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_model_dir(path: Path) -> bool:
    return (path / "meta.json").exists() and (path / "preprocessor.joblib").exists() and (path / "model.pt").exists()


def _discover_model_dirs() -> list[str]:
    root = _project_root()
    outputs = root / "outputs"
    if not outputs.exists():
        return []

    candidates: list[Path] = []
    # Common layout: outputs/model and outputs/<name>/model
    for p in outputs.rglob("model"):
        if p.is_dir() and _is_model_dir(p):
            candidates.append(p)

    # De-duplicate and sort for stable UI.
    uniq = sorted({str(p) for p in candidates})
    return uniq


def _settings_path() -> Path:
    return _project_root() / ".streamlit" / "email_privilege_dashboard_settings.json"


def _load_settings() -> dict[str, Any]:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(settings: dict[str, Any]) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def _discover_report_dirs() -> list[str]:
    root = _project_root()
    outputs = root / "outputs"
    candidates: list[Path] = []

    default_reports = root / "outputs" / "reports"
    if default_reports.exists() and default_reports.is_dir():
        candidates.append(default_reports)

    if outputs.exists():
        for p in outputs.rglob("reports"):
            if p.is_dir():
                candidates.append(p)

    uniq = sorted({str(p) for p in candidates})
    return uniq


@st.cache_resource(show_spinner=False)
def load_artifacts(model_dir: str) -> tuple[dict[str, Any], Any, GRUClassifier]:
    md = Path(model_dir)
    meta_path = md / "meta.json"
    preproc_path = md / "preprocessor.joblib"
    model_path = md / "model.pt"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing model metadata: {meta_path}")
    if not preproc_path.exists():
        raise FileNotFoundError(f"Missing preprocessor: {preproc_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights: {model_path}")

    meta = _read_json(meta_path)
    train_cfg = meta.get("config", {}).get("train", {})

    preprocessor = joblib.load(preproc_path)

    # Infer input_size robustly using the columns the preprocessor was fit on.
    feature_names_in = getattr(preprocessor, "feature_names_in_", None)
    if feature_names_in is None:
        raise RuntimeError(
            "Preprocessor is missing feature_names_in_. Re-train the model to regenerate artifacts."
        )

    dummy = pd.DataFrame([{c: 0 for c in feature_names_in}])
    X0 = preprocessor.transform(dummy)
    input_size = int(X0.shape[1])

    model = GRUClassifier(
        input_size=input_size,
        hidden_size=int(train_cfg.get("hidden_size", 64)),
        num_layers=int(train_cfg.get("num_layers", 1)),
        dropout=float(train_cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return meta, preprocessor, model


def build_single_sequence(x_evt: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a single end-anchored (left-padded) sequence from one event."""
    if x_evt.ndim != 2 or x_evt.shape[0] != 1:
        raise ValueError("x_evt must have shape (1, n_features)")
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")

    pad_len = max(0, seq_len - 1)
    if pad_len:
        pad = np.zeros((pad_len, x_evt.shape[1]), dtype=np.float32)
        X_seq = np.vstack([pad, x_evt.astype(np.float32)])
    else:
        X_seq = x_evt.astype(np.float32)

    X_seq = X_seq.reshape(1, seq_len, -1)
    lengths = np.asarray([1], dtype=np.int64)
    return X_seq, lengths


def _build_download_report(
    *,
    probability: float,
    threshold: float,
    is_alert: bool,
    raw_event: dict[str, Any],
    feature_mode: str,
    feature_cols: list[str],
    engineered_values: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    cfg = meta.get("config", {}) if isinstance(meta, dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    return {
        "task": "email_privilege_misuse_detection",
        "probability": float(probability),
        "threshold": float(threshold),
        "is_alert": bool(is_alert),
        "raw_event": raw_event,
        "feature_mode": feature_mode,
        "feature_cols": feature_cols,
        "engineered_feature_values": engineered_values,
        "model": {
            "hidden_size": train_cfg.get("hidden_size"),
            "num_layers": train_cfg.get("num_layers"),
            "dropout": train_cfg.get("dropout"),
            "seq_len": train_cfg.get("seq_len"),
        },
    }


def main() -> None:
    st.title("Email Privilege Misuse Detector")
    st.caption("Score a single email event using the trained GRU model artifacts.")
    st.divider()

    st.markdown(
        """
        <style>
        /* Rounded inputs + cleaner spacing */
        .stTextInput input, .stNumberInput input, .stTextArea textarea {
            border-radius: 10px;
        }
        /* Round selectbox / multiselect containers */
        div[data-baseweb="select"] > div {
            border-radius: 10px;
        }
        /* Slightly tighter header spacing */
        h1, h2, h3 { letter-spacing: 0.2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    out_cfg = OutputConfig()

    settings = _load_settings()
    default_report_dir = str(settings.get("report_dir") or (_project_root() / "outputs" / "reports"))

    st.sidebar.header("Model")

    discovered = _discover_model_dirs()
    default_model_dir = str(out_cfg.model_dir)
    model_dir: str

    if discovered:
        options = discovered
        if default_model_dir not in options:
            options = [default_model_dir] + options

        try:
            default_index = options.index(default_model_dir)
        except ValueError:
            default_index = 0

        picked_model_dir = st.sidebar.selectbox(
            "Model directory (quick pick)",
            options=options,
            index=default_index,
            help="Pick a discovered model directory, or edit the path below.",
        )
        model_dir = st.sidebar.text_input(
            "Model directory path",
            value=picked_model_dir,
            help="You can type/paste a custom path here.",
        )
    else:
        model_dir = st.sidebar.text_input(
            "Model directory path",
            value=default_model_dir,
            help="Type/paste the path that contains meta.json, preprocessor.joblib, and model.pt.",
        )

    model_available = True
    meta: dict[str, Any] = {}
    preprocessor: Any | None = None
    model: GRUClassifier | None = None

    try:
        meta, preprocessor, model = load_artifacts(model_dir)
        st.sidebar.success("Artifacts loaded")
    except Exception as e:
        model_available = False
        st.sidebar.warning("Model artifacts not available — using simulated score")
        st.sidebar.caption(str(e))

    default_seq_len = 20
    if model_available:
        train_cfg = meta.get("config", {}).get("train", {})
        default_seq_len = int(train_cfg.get("seq_len", 20))

    st.sidebar.header("Scoring")
    st.sidebar.write(f"Decision threshold: **{_FIXED_THRESHOLD:.2f}**")
    seq_len = st.sidebar.number_input(
        "Sequence length (padding)",
        min_value=2,
        max_value=200,
        value=int(default_seq_len),
        step=1,
        help="The GRU was trained on end-anchored sequences; a single email is left-padded to this length.",
    )

    if model_available:
        with st.sidebar.expander("Model details", expanded=False):
            data_cfg = meta.get("config", {}).get("data", {})
            st.write(
                {
                    "feature_mode": data_cfg.get("feature_mode", "minimal"),
                    "timestamp_col": data_cfg.get("timestamp_col", "date"),
                    "id_col": data_cfg.get("id_col", "id"),
                    "group_cols": data_cfg.get("group_col_candidates", ["user", "from"]),
                }
            )

    st.sidebar.header("Risk Scoring Configuration")
    scoring_mode = st.sidebar.selectbox(
        "Scoring mode",
        options=["Model Prediction", "Manual Scoring", "Hybrid"],
        index=0,
        help="Model Prediction uses the GRU output. Manual Scoring uses configurable rules. Hybrid blends both.",
    )

    with st.sidebar.expander("Manual weights", expanded=True):
        after_hours_weight = st.slider("After-hours email weight", 0.0, 1.0, 0.20, 0.01)
        large_content_weight = st.slider("Large content size weight", 0.0, 1.0, 0.20, 0.01)
        attachment_weight = st.slider("Attachment weight", 0.0, 1.0, 0.15, 0.01)
        external_recipient_weight = st.slider("External recipient weight", 0.0, 1.0, 0.15, 0.01)
        suspicious_keyword_weight = st.slider("Suspicious keywords weight", 0.0, 1.0, 0.20, 0.01)

    with st.sidebar.expander("Rule thresholds", expanded=True):
        bh_start = st.number_input("Business hours start", min_value=0, max_value=23, value=8, step=1)
        bh_end = st.number_input("Business hours end", min_value=1, max_value=24, value=18, step=1)
        content_size_threshold = st.number_input(
            "Content size threshold (bytes)",
            min_value=0,
            value=1_000_000,
            step=500,
            help="Large content is scored once size exceeds this threshold.",
        )
        attachment_threshold = st.number_input(
            "Attachment count threshold",
            min_value=0,
            value=3,
            step=1,
            help="Too many attachments is scored once attachments exceed this threshold.",
        )

    suspicious_keywords_csv = st.sidebar.text_input(
        "Suspicious keywords (comma-separated)",
        value="password, confidential, privileged, admin",
        help="Case-insensitive substring match against the email content.",
    )

    st.sidebar.subheader("Hybrid Blending Configuration")
    hybrid_disabled = scoring_mode != "Hybrid"

    _preset_to_weight = {
        "Balanced (0.50 / 0.50)": 0.50,
        "Model-Focused (0.70 / 0.30)": 0.70,
        "Manual-Focused (0.30 / 0.70)": 0.30,
        "Custom": None,
    }

    if "hybrid_model_weight" not in st.session_state:
        st.session_state["hybrid_model_weight"] = 0.50
    if "hybrid_blend_preset" not in st.session_state:
        st.session_state["hybrid_blend_preset"] = "Balanced (0.50 / 0.50)"

    preset_choice = st.sidebar.selectbox(
        "Quick preset",
        options=list(_preset_to_weight.keys()),
        key="hybrid_blend_preset",
        disabled=hybrid_disabled,
    )
    preset_weight = _preset_to_weight.get(str(preset_choice))
    if preset_weight is not None:
        st.session_state["hybrid_model_weight"] = float(preset_weight)

    model_weight = st.sidebar.slider(
        "Model weight",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["hybrid_model_weight"]),
        step=0.05,
        key="hybrid_model_weight",
        disabled=hybrid_disabled,
    )

    # If the user drifts from a preset value, reflect that as "Custom".
    if not hybrid_disabled:
        active_preset_weight = _preset_to_weight.get(str(st.session_state.get("hybrid_blend_preset")))
        if active_preset_weight is not None and abs(float(model_weight) - float(active_preset_weight)) > 1e-9:
            st.session_state["hybrid_blend_preset"] = "Custom"

    model_weight = _clamp01(float(model_weight))
    manual_weight = _clamp01(1.0 - float(model_weight))

    st.sidebar.write(f"Model weight: **{model_weight:.2f}**")
    st.sidebar.write(f"Manual weight: **{manual_weight:.2f}**")
    st.sidebar.caption(f"Final score = {model_weight:.2f} × model + {manual_weight:.2f} × manual")
    st.sidebar.caption(
        "Higher model weight increases the influence of the ML prediction. "
        "Higher manual weight increases the influence of rule-based scoring."
    )

    # Keep backwards-compatible name for downstream config/persistence.
    hybrid_alpha = float(model_weight)

    risk_cfg = ScoringConfig(
        threshold=float(_FIXED_THRESHOLD),
        scoring_mode=str(scoring_mode),
        hybrid_alpha=float(hybrid_alpha),
        base_risk_score=0.10,
        after_hours_weight=float(after_hours_weight),
        large_content_weight=float(large_content_weight),
        attachment_weight=float(attachment_weight),
        external_recipient_weight=float(external_recipient_weight),
        suspicious_keyword_weight=float(suspicious_keyword_weight),
        weekend_login_weight=ScoringConfig.weekend_login_weight,
        high_login_frequency_weight=ScoringConfig.high_login_frequency_weight,
        unusual_device_weight=ScoringConfig.unusual_device_weight,
        content_size_threshold=int(content_size_threshold),
        attachment_threshold=int(attachment_threshold),
        login_frequency_threshold=ScoringConfig.login_frequency_threshold,
        business_hours_start=int(bh_start),
        business_hours_end=int(bh_end),
        suspicious_keywords=_parse_keywords(str(suspicious_keywords_csv)),
    )

    st.sidebar.header("Report output")
    discovered_reports = _discover_report_dirs()
    report_dir_options = discovered_reports
    if default_report_dir not in report_dir_options:
        report_dir_options = [default_report_dir] + report_dir_options

    if report_dir_options:
        try:
            default_reports_index = report_dir_options.index(default_report_dir)
        except ValueError:
            default_reports_index = 0
        picked_report_dir = st.sidebar.selectbox(
            "Save directory (quick pick)",
            options=report_dir_options,
            index=default_reports_index,
            help="Pick a discovered reports directory, or edit the path below.",
        )
        selected_report_dir = st.sidebar.text_input(
            "Save directory path",
            value=picked_report_dir,
            help="Directory where JSON reports are written after scoring.",
        )
    else:
        selected_report_dir = st.sidebar.text_input(
            "Save directory path",
            value=default_report_dir,
            help="Directory where JSON reports are written after scoring.",
        )

    save_to_disk = st.sidebar.checkbox("Also save report to disk", value=True)

    # Persist the chosen report dir across sessions.
    prev_report_dir = str(settings.get("report_dir") or "")
    if selected_report_dir and selected_report_dir != prev_report_dir:
        settings["report_dir"] = selected_report_dir
        _save_settings(settings)

    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.subheader("Input")
        st.write("Enter one email event. The model predicts whether it resembles overused privileges/risky behavior.")

        with st.form("email_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                sender = st.text_input("From (sender)", value="user@company.com")
            with col2:
                recipient = st.text_input("To (recipient)", value="recipient@company.com")

            now = pd.Timestamp.utcnow().to_pydatetime()
            event_time = st.date_input("Shared time (date)", value=now.date())
            event_clock = st.time_input("Shared time (time)", value=now.time().replace(microsecond=0))

            col3, col4 = st.columns(2)
            with col3:
                size = st.number_input(
                    "Size of content shared (bytes)",
                    min_value=0,
                    value=0,
                    step=1,
                )
            with col4:
                attachments = st.number_input(
                    "Number of attachments",
                    min_value=0,
                    value=0,
                    step=1,
                )

            content = st.text_area("Email content", value="", height=160, placeholder="Paste email body here...")
            submitted = st.form_submit_button("Detect")

    with right:
        st.subheader("Results")
        if "last_score" not in st.session_state:
            st.info("Run detection to see the score and report.")
        else:
            last = st.session_state["last_score"]
            st.metric("Final risk score", f"{last['final_score']:.3f}")
            st.write(
                f"Mode: **{last['scoring_mode']}**  \\  Decision (threshold={last['threshold']:.2f}): **{last['decision']}**"
            )
            if last["is_alert"]:
                st.error("Alert: this email looks risky under the configured scoring.")
            else:
                st.success("No alert: this email looks normal under the configured scoring.")

            cols = st.columns(2)
            cols[0].metric("Model probability", f"{last['model_probability']:.3f}")
            cols[1].metric("Manual score", f"{last['manual_score']:.3f}")

            with st.expander("Rule-by-rule explanation", expanded=True):
                st.dataframe(pd.DataFrame(last.get("rule_breakdown", [])), use_container_width=True)

            saved_path = last.get("saved_report_path")
            save_error = last.get("save_error")
            if saved_path:
                st.caption(f"Saved report: {saved_path}")
            elif save_error:
                st.warning(f"Could not save report to disk: {save_error}")

            st.download_button(
                "Download JSON report",
                data=json.dumps(last["report"], indent=2, ensure_ascii=False),
                file_name="email_privilege_detection_report.json",
                mime="application/json",
                use_container_width=True,
            )

            with st.expander("Inputs", expanded=False):
                st.write(last["raw_event"])

            with st.expander("Engineered features used", expanded=False):
                st.code(", ".join(last["feature_cols"]), language="text")
                if last["engineered_feature_values"]:
                    st.dataframe(
                        pd.DataFrame([last["engineered_feature_values"]]),
                        use_container_width=True,
                    )

    if not submitted:
        return

    if not sender.strip():
        st.warning("Please enter a sender.")
        return

    dt = pd.Timestamp.combine(event_time, event_clock)

    # Build a single-row dataframe in the same schema the training pipeline expects.
    raw_df = pd.DataFrame(
        [
            {
                "id": "manual_input",
                "date": dt,
                "to": recipient,
                "from": sender,
                "size": int(size),
                "attachments": int(attachments),
                "content": content,
            }
        ]
    )

    raw_event = {
        "id": "manual_input",
        "date": str(dt),
        "to": recipient,
        "from": sender,
        "size": int(size),
        "attachments": int(attachments),
        "content": content,
    }

    feature_mode = "minimal"
    feature_cols: list[str] = []
    engineered_values: dict[str, Any] = {}

    with st.spinner("Scoring..."):
        if model_available:
            # Use the same feature engineering as training (minimal mode by default in meta.json).
            data_cfg = meta.get("config", {}).get("data", {})
            feature_mode = str(data_cfg.get("feature_mode", "minimal"))

            prep = prepare_dataframe(
                raw_df,
                timestamp_col=str(data_cfg.get("timestamp_col", "date")),
                id_col=str(data_cfg.get("id_col", "id")),
                group_col_candidates=tuple(data_cfg.get("group_col_candidates", ["user", "from"])),
                label_col=None,
                feature_mode=feature_mode,
            )

            try:
                assert preprocessor is not None
                X_evt = preprocessor.transform(prep.df[prep.feature_cols])
            except Exception as e:
                st.error(
                    "Input could not be transformed by the trained preprocessor. "
                    "This usually means the model was trained with different features.\n\n"
                    f"Details: {e}"
                )
                st.stop()

            X_seq, lengths = build_single_sequence(X_evt, int(seq_len))

            X_t = torch.tensor(X_seq, dtype=torch.float32)
            L_t = torch.tensor(lengths, dtype=torch.int64)
            assert model is not None
            logit = model(X_t, L_t)
            model_prob = float(torch.sigmoid(logit).detach().cpu().item())

            feature_cols = list(prep.feature_cols)
            show_cols = [c for c in prep.feature_cols if c in prep.df.columns]
            if show_cols:
                engineered_values = prep.df.loc[:, show_cols].iloc[0].to_dict()
        else:
            # Simulated “GRU prediction”: random probability in [0, 1].
            model_prob = float(random.random())

        manual_score, rule_breakdown = _manual_score_email(
            sender=sender,
            recipient=recipient,
            dt=pd.Timestamp(dt),
            size_bytes=int(size),
            attachments=int(attachments),
            content=str(content),
            cfg=risk_cfg,
        )

        final_score, is_alert, decision = _final_score_and_decision(
            model_probability=float(model_prob),
            manual_score=float(manual_score),
            cfg=risk_cfg,
            threshold=float(_FIXED_THRESHOLD),
        )

    threshold = float(_FIXED_THRESHOLD)

    report = _build_download_report(
        probability=final_score,
        threshold=threshold,
        is_alert=is_alert,
        raw_event=raw_event,
        feature_mode=feature_mode,
        feature_cols=feature_cols,
        engineered_values=engineered_values,
        meta=meta,
    )

    # Attach scoring engine details to the report (keeps report download behavior intact).
    report["risk_scoring"] = {
        "mode": risk_cfg.scoring_mode,
        "final_score": float(final_score),
        "model_probability": float(model_prob),
        "manual_score": float(manual_score),
        "hybrid_alpha": float(risk_cfg.hybrid_alpha),
        "rule_breakdown": rule_breakdown,
        "config": asdict(risk_cfg),
    }

    st.session_state["last_score"] = {
        "final_score": float(final_score),
        "model_probability": float(model_prob),
        "manual_score": float(manual_score),
        "threshold": threshold,
        "is_alert": is_alert,
        "decision": decision,
        "scoring_mode": risk_cfg.scoring_mode,
        "rule_breakdown": rule_breakdown,
        "raw_event": raw_event,
        "feature_cols": feature_cols,
        "engineered_feature_values": engineered_values,
        "report": report,
    }

    if save_to_disk:
        try:
            report_dir_path = Path(selected_report_dir)
            report_dir_path.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir_path / f"email_privilege_detection_report_{ts}.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            st.session_state["last_score"]["saved_report_path"] = str(report_path)
        except Exception as e:
            st.session_state["last_score"]["saved_report_path"] = None
            st.session_state["last_score"]["save_error"] = str(e)

    st.rerun()


if __name__ == "__main__":
    main()
