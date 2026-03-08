"""Local alert integrity verification (JSONL ledger).

This module makes alerts *tamper-evident* by:
- generating a deterministic canonical JSON representation of each alert
- hashing that canonical form (SHA-256 by default)
- appending the hash to a local JSONL ledger
- verifying later whether an alert has changed

It is intentionally simple and suitable for a final-year research project.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(obj: Any, exclude_fields: set[str]) -> Any:
    """Convert an object into a JSON-serializable structure.

    - Dict keys become strings.
    - Dict keys listed in exclude_fields are removed (at any nesting level).
    - Lists preserve order.
    - Datetimes are converted to ISO-8601 strings.
    - Unknown objects fall back to str(obj).
    """

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if ks in exclude_fields:
                continue
            out[ks] = _sanitize(v, exclude_fields)
        return out

    if isinstance(obj, list):
        return [_sanitize(v, exclude_fields) for v in obj]

    if isinstance(obj, tuple):
        return [_sanitize(v, exclude_fields) for v in obj]

    if isinstance(obj, datetime):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Keep deterministic by converting unknown objects.
    return str(obj)


def canonicalize_alert(alert: dict, exclude_fields: list[str] | None = None) -> str:
    """Return a deterministic JSON string representation of an alert.

    Args:
        alert: Alert record dict.
        exclude_fields: Optional list of field names to exclude from hashing.
            Exclusion is applied to dict keys at any nesting level.

    Returns:
        Canonical JSON string (sorted keys, compact separators).

    Raises:
        ValueError: if alert is not a dict.
    """

    if not isinstance(alert, dict):
        raise ValueError("alert must be a dict")

    exclude = set(str(x) for x in (exclude_fields or []))
    safe = _sanitize(alert, exclude)
    return json.dumps(safe, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_alert_hash(
    alert: dict,
    hash_algorithm: str = "sha256",
    exclude_fields: list[str] | None = None,
) -> str:
    """Compute a cryptographic hash of an alert's canonicalized form.

    Args:
        alert: Alert dict.
        hash_algorithm: Hashlib algorithm name (default: sha256).
        exclude_fields: Optional list of excluded keys for canonicalization.

    Returns:
        Hex digest string.
    """

    algo = (hash_algorithm or "sha256").strip().lower()
    payload = canonicalize_alert(alert, exclude_fields=exclude_fields).encode("utf-8")
    h = hashlib.new(algo)
    h.update(payload)
    return h.hexdigest()


def append_to_integrity_ledger(alert: dict, hash_value: str, ledger_path: str | Path) -> dict:
    """Append a single integrity record to a JSONL ledger.

    Ledger record fields:
        - alert_id
        - timestamp
        - event_type
        - final_score
        - decision
        - hash
    """

    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "alert_id": None if alert.get("alert_id") is None else str(alert.get("alert_id")),
        "timestamp": _now_iso(),
        "event_type": alert.get("event_type"),
        "final_score": alert.get("final_score"),
        "decision": alert.get("decision"),
        "hash": str(hash_value),
    }

    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_sanitize(record, exclude_fields=set()), ensure_ascii=False) + "\n")

    return record


def load_integrity_ledger(ledger_path: str | Path) -> list[dict]:
    """Load all ledger entries from a JSONL file.

    Returns an empty list if the file is missing.
    """

    p = Path(ledger_path)
    if not p.exists() or not p.is_file():
        return []

    out: list[dict] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def find_ledger_record(alert_id: str, ledger_path: str | Path) -> dict | None:
    """Find a ledger entry for a given alert_id."""

    target = str(alert_id)
    for rec in load_integrity_ledger(ledger_path):
        if str(rec.get("alert_id")) == target:
            return rec
    return None


def verify_alert_hash(
    alert: dict,
    ledger_path: str | Path,
    hash_algorithm: str = "sha256",
    exclude_fields: list[str] | None = None,
) -> dict:
    """Verify an alert record against a stored ledger hash.

    Returns a dict with:
        - alert_id
        - status: "Verified" | "Tampered" | "Missing"
        - stored_hash
        - computed_hash
        - matched (bool)
    """

    alert_id = None if alert.get("alert_id") is None else str(alert.get("alert_id"))
    computed = compute_alert_hash(alert, hash_algorithm=hash_algorithm, exclude_fields=exclude_fields)

    if alert_id is None:
        return {
            "alert_id": None,
            "status": "Missing",
            "stored_hash": None,
            "computed_hash": computed,
            "matched": False,
        }

    rec = find_ledger_record(alert_id, ledger_path)
    if rec is None:
        return {
            "alert_id": alert_id,
            "status": "Missing",
            "stored_hash": None,
            "computed_hash": computed,
            "matched": False,
        }

    stored = rec.get("hash")
    matched = str(stored) == str(computed)
    return {
        "alert_id": alert_id,
        "status": "Verified" if matched else "Tampered",
        "stored_hash": stored,
        "computed_hash": computed,
        "matched": bool(matched),
    }


def register_alert_integrity(
    alert: dict,
    ledger_path: str | Path,
    hash_algorithm: str = "sha256",
    exclude_fields: list[str] | None = None,
) -> dict:
    """Convenience function: compute hash and append ledger entry."""

    h = compute_alert_hash(alert, hash_algorithm=hash_algorithm, exclude_fields=exclude_fields)
    ledger_record = append_to_integrity_ledger(alert, h, ledger_path)
    return {"hash": h, "ledger_record": ledger_record}


if __name__ == "__main__":
    demo_ledger = Path("outputs") / "integrity" / "ledger_demo.jsonl"
    if demo_ledger.exists():
        demo_ledger.unlink()

    demo_alert = {
        "alert_id": "demo-001",
        "event_type": "email",
        "final_score": 0.72,
        "confidence_label": "high",
        "decision": "ALERT",
        "processed_at": "2026-03-07T12:00:00+00:00",
        "cleaned_inputs": {"sender": "alice@company.com", "recipient": "bob@external.com"},
        "engineered_features": {"after_hours": True, "attachment_count": 4},
        "triggered_rules": ["after_hours", "attachments"],
    }

    print("Registering integrity...")
    reg = register_alert_integrity(demo_alert, demo_ledger)
    print(reg)

    print("Verifying (expected Verified)...")
    print(verify_alert_hash(demo_alert, demo_ledger))

    tampered = dict(demo_alert)
    tampered["final_score"] = 0.10

    print("Verifying tampered (expected Tampered)...")
    print(verify_alert_hash(tampered, demo_ledger))
