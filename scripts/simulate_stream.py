from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import itertools
from pathlib import Path
import sys
from typing import Any, Dict, Literal

from insider_gru.config import ScoringConfig, get_default_config
from insider_gru.integrity import register_alert_integrity
from insider_gru.pipeline import run_detection_pipeline
from insider_gru.streaming import (
    clear_output_file,
    stream_events_from_csv,
    write_live_alert,
    write_live_event,
)


EventType = Literal["email", "login"]


def _safe_print(*args: Any, **kwargs: Any) -> None:
    """Print without failing the whole run if the stdout pipe closes early."""

    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Simulate near-real-time streaming from a CSV, run the single-event pipeline, "
            "and append events/alerts to JSONL outputs."
        )
    )
    p.add_argument(
        "--event-type",
        type=str,
        choices=["email", "login"],
        default="email",
        help="Which event type to stream (default: email)",
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to a CSV event file (e.g., Dataset/train/test/R2/email-train-data.csv)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds to wait between events (default: 0.25)",
    )
    p.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing JSONL outputs before streaming",
    )
    return p.parse_args()


def _build_pipeline_config() -> Dict[str, Any]:
    """Build the pipeline's free-form config dict from the default dataclasses."""

    app_cfg = get_default_config()
    scoring: ScoringConfig = app_cfg.scoring

    # pipeline._parse_config expects ScoringConfig fields at top level.
    cfg: Dict[str, Any] = {**asdict(scoring)}

    # Use the repo's canonical timestamp column from DataConfig by default.
    # For the provided CERT-style CSVs, this is typically 'date'.
    cfg["email_timestamp_col"] = str(app_cfg.data.timestamp_col)
    cfg["login_timestamp_col"] = str(app_cfg.data.timestamp_col)
    return cfg


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _format_timestamp(value: Any) -> str:
    """Best-effort timestamp formatting for console summaries."""

    if value is None:
        return "-"
    if isinstance(value, datetime):
        try:
            return value.isoformat(sep=" ", timespec="seconds")
        except Exception:
            return str(value)
    s = str(value).strip()
    return s if s else "-"


def main() -> None:
    args = parse_args()
    event_type: EventType = str(args.event_type)

    app_cfg = get_default_config()
    out_dir = Path(app_cfg.output.out_dir) / "live"
    live_events_path = out_dir / "live_events.jsonl"
    live_alerts_path = out_dir / "live_alerts.jsonl"

    integrity_enabled = bool(getattr(app_cfg.integrity, "enabled", False))
    ledger_path = Path(getattr(app_cfg.integrity, "ledger_path", Path("outputs") / "integrity" / "ledger.jsonl"))
    hash_algorithm = str(getattr(app_cfg.integrity, "hash_algorithm", "sha256"))

    if args.clear_output:
        clear_output_file(live_events_path)
        clear_output_file(live_alerts_path)

    pipeline_cfg = _build_pipeline_config()

    source_path = Path(args.source)
    processed = 0
    alerts = 0
    skipped = 0

    _safe_print(f"Event type: {event_type}")
    _safe_print(f"Source CSV: {source_path}")
    _safe_print(f"Delay (s): {float(args.delay)}")
    _safe_print(f"Events JSONL: {live_events_path}")
    _safe_print(f"Alerts JSONL: {live_alerts_path}")
    if integrity_enabled:
        _safe_print(f"Integrity ledger: {ledger_path} ({hash_algorithm})")
    else:
        _safe_print("Integrity ledger: disabled")
    _safe_print("---")

    ts_key = pipeline_cfg["email_timestamp_col"] if event_type == "email" else pipeline_cfg["login_timestamp_col"]

    # Validate loading early (generator body runs on first next()).
    try:
        base_iter = stream_events_from_csv(source_path, delay_seconds=float(args.delay))
        first_event = next(iter(base_iter))
        event_iter = itertools.chain([first_event], base_iter)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"Failed to load source CSV: {e}") from e
    except StopIteration:
        raise SystemExit(f"Source CSV produced no rows: {source_path}")

    for event in event_iter:
        # Write raw event first so dashboards can show the live feed even if processing fails.
        try:
            write_live_event(event, live_events_path)
        except Exception as e:
            skipped += 1
            _safe_print(f"[skip] failed writing event JSONL: {e}")
            continue

        try:
            alert = run_detection_pipeline(
                raw_event=event,
                event_type=event_type,
                config=pipeline_cfg,
                model_score=None,
            )
        except Exception as e:
            skipped += 1
            _safe_print(f"[skip] pipeline error: {e}")
            continue

        integrity_note = "disabled"
        if integrity_enabled:
            try:
                # Exclude any integrity fields to avoid hashing self-referential additions.
                reg = register_alert_integrity(
                    alert,
                    ledger_path=ledger_path,
                    hash_algorithm=hash_algorithm,
                    exclude_fields=["integrity_hash"],
                )
                integrity_hash = str(reg.get("hash"))
                # Attach hash into the alert for downstream dashboards/reports.
                alert["integrity_hash"] = integrity_hash
                integrity_note = integrity_hash[:8]
            except Exception as e:
                # Keep pipeline flow robust: write alert even if integrity registration fails.
                integrity_note = f"error:{type(e).__name__}"

        try:
            write_live_alert(alert, live_alerts_path)
        except Exception as e:
            skipped += 1
            _safe_print(f"[skip] failed writing alert JSONL: {e}")
            continue

        processed += 1
        if alert.get("decision") == "ALERT":
            alerts += 1

        # Clean per-event console summary
        ts_val = event.get(ts_key)
        alert_id = str(alert.get("alert_id") or "-")
        final_score = _safe_float(alert.get("final_score"))
        conf = str(alert.get("confidence_label", "-") or "-")
        decision = str(alert.get("decision", "-") or "-")
        score_str = "-" if final_score is None else f"{final_score:.3f}"

        _safe_print(
            f"{_format_timestamp(ts_val)} | {event_type:<5} | id={alert_id} | score={score_str} | "
            f"confidence={conf:<6} | decision={decision} | integrity={integrity_note}"
        )

    _safe_print("---")
    _safe_print(f"Processed events: {processed}")
    _safe_print(f"ALERT decisions: {alerts}")
    _safe_print(f"Skipped rows: {skipped}")
    _safe_print(f"Events JSONL: {live_events_path}")
    _safe_print(f"Alerts JSONL: {live_alerts_path}")


if __name__ == "__main__":
    main()
