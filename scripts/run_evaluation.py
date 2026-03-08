from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_json_safe(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.ndarray):
        return [_to_json_safe(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _to_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_json_safe(v) for v in x]
    return str(x)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json_safe(obj), indent=2), encoding="utf-8")


def _extract_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize metric names into a standard set."""

    out = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
    }
    if not metrics:
        return out

    for k in out.keys():
        if k in metrics:
            out[k] = metrics.get(k)
    return out


def _extract_notes(metrics: dict[str, Any] | None) -> str:
    if not metrics or not isinstance(metrics, dict):
        return ""
    parts: list[str] = []
    notes = metrics.get("notes")
    if isinstance(notes, str) and notes.strip():
        parts.append(notes.strip())

    metric_notes = metrics.get("metric_notes")
    if isinstance(metric_notes, dict):
        roc_note = metric_notes.get("roc_auc")
        if isinstance(roc_note, str) and roc_note.strip():
            parts.append(roc_note.strip())
        err_note = metric_notes.get("error")
        if isinstance(err_note, str) and err_note.strip():
            parts.append(err_note.strip())

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return "; ".join(unique)


def _load_baselines_metrics(path: Path) -> dict[str, dict[str, Any]]:
    """Load baselines metrics.json which is stored as a dict keyed by model name."""

    if not path.exists():
        return {}
    data = _read_json(path)
    if not isinstance(data, dict):
        return {}
    # Expected keys: logistic_regression/random_forest/isolation_forest
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def _load_baselines_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = _read_json(path)
    return data if isinstance(data, dict) else {}


def _fmt_float(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if not np.isfinite(v):
            return ""
        return f"{v:.3f}"
    if isinstance(v, (np.floating,)):
        return _fmt_float(float(v))
    return str(v)


def main() -> None:
    # Inputs (required locations)
    baselines_metrics_path = Path("outputs") / "baselines" / "login" / "metrics.json"
    if not baselines_metrics_path.exists():
        baselines_metrics_path = Path("outputs") / "baselines" / "email" / "metrics.json"

    baselines_root = baselines_metrics_path.parent
    baselines_meta = _load_baselines_meta(baselines_root / "meta.json")
    baselines_status = baselines_meta.get("model_status") if isinstance(baselines_meta, dict) else None
    if not isinstance(baselines_status, dict):
        baselines_status = {}

    gru_metrics_path = Path("outputs") / "reports" / "metrics.json"
    transformer_metrics_path = Path("outputs") / "transformer" / "reports" / "metrics.json"

    # Load
    baselines = _load_baselines_metrics(baselines_metrics_path)
    gru_metrics = _read_json(gru_metrics_path) if gru_metrics_path.exists() else None
    transformer_metrics = _read_json(transformer_metrics_path) if transformer_metrics_path.exists() else None

    # Build comparison table (exact required rows)
    model_rows: list[dict[str, Any]] = []
    model_order = [
        ("Logistic Regression", "logistic_regression"),
        ("Random Forest", "random_forest"),
        ("Isolation Forest", "isolation_forest"),
        ("GRU", "gru"),
        ("Transformer", "transformer"),
    ]

    for display_name, key in model_order:
        if key in {"gru", "transformer"}:
            m = gru_metrics if key == "gru" else transformer_metrics
            metrics = _extract_metrics(m if isinstance(m, dict) else None)
            notes = _extract_notes(m if isinstance(m, dict) else None)
        else:
            bm = baselines.get(key)
            metrics = _extract_metrics(bm)
            notes = _extract_notes(bm)
            if not notes:
                status = baselines_status.get(key)
                if isinstance(status, str) and status.strip():
                    notes = f"baseline status: {status.strip()}"
                elif bm is None:
                    notes = "baseline metrics missing (model key not present in metrics.json)"

        model_rows.append({"model": display_name, **metrics, "notes": notes})

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comparison_df = pd.DataFrame(model_rows, columns=["model", *metric_cols, "notes"])

    # Outputs (required)
    out_dir = Path("outputs") / "final_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export CSV without literal NaN values (keep blanks for unavailable metrics).
    export_df = comparison_df.copy()
    for c in metric_cols:
        export_df[c] = export_df[c].apply(
            lambda v: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else v
        )
    export_df.to_csv(out_dir / "model_comparison.csv", index=False, na_rep="")

    summary = {row["model"]: {k: row.get(k) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]} for row in model_rows}
    _save_json(out_dir / "summary_metrics.json", summary)

    # Plots (required)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.insider_gru.plots import plot_model_comparison_bar
    except ModuleNotFoundError:
        from insider_gru.plots import plot_model_comparison_bar

    if "f1" in comparison_df.columns:
        plot_model_comparison_bar(
            comparison_df,
            metric="f1",
            output_path=plots_dir / "model_comparison_f1.png",
            title="Model Comparison (F1)",
        )
    if "roc_auc" in comparison_df.columns:
        plot_model_comparison_bar(
            comparison_df,
            metric="roc_auc",
            output_path=plots_dir / "model_comparison_roc_auc.png",
            title="Model Comparison (ROC-AUC)",
        )

    # Terminal output
    print(f"[run_evaluation] wrote: {out_dir}")
    print(export_df.to_string(index=False, formatters={
        "accuracy": _fmt_float,
        "precision": _fmt_float,
        "recall": _fmt_float,
        "f1": _fmt_float,
        "roc_auc": _fmt_float,
    }))


if __name__ == "__main__":
    main()
