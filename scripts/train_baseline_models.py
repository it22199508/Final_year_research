from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

try:
    from src.insider_gru.baselines import (
        evaluate_predictions,
        predict_isolation_forest_scores,
        predict_supervised_probabilities,
        train_isolation_forest,
        train_logistic_regression,
        train_random_forest,
    )
    from src.insider_gru.config import config_to_dict, get_default_config
    from src.insider_gru.data import (
        attach_labels,
        discover_group_folders,
        load_train_test_event_logs,
        prepare_email_training_data,
        prepare_login_training_data,
        resolve_train_test_split_dirs,
    )
except ModuleNotFoundError:
    # Fallback for editable/installed package layouts where the import root is `insider_gru`.
    from insider_gru.baselines import (
        evaluate_predictions,
        predict_isolation_forest_scores,
        predict_supervised_probabilities,
        train_isolation_forest,
        train_logistic_regression,
        train_random_forest,
    )
    from insider_gru.config import config_to_dict, get_default_config
    from insider_gru.data import (
        attach_labels,
        discover_group_folders,
        load_train_test_event_logs,
        prepare_email_training_data,
        prepare_login_training_data,
        resolve_train_test_split_dirs,
    )


EventType = Literal["email", "device", "http", "login"]


def _canonical_event_type(x: str) -> str:
    s = str(x).strip().lower()
    return "device" if s == "login" else s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train classic baseline models (Logistic Regression / Random Forest / Isolation Forest) "
            "on fixed-length sequences for research comparison."
        )
    )
    p.add_argument(
        "--event-type",
        type=str,
        choices=["email", "device", "http", "login"],
        default="email",
        help="Which event type to train on (default: email). 'login' is an alias for 'device'.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for evaluation (default: config scoring threshold)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/baselines/<event_type>/)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose dataset/label debug prints during preparation",
    )
    return p.parse_args()
def _pick_label_col(cfg: Any, df_columns: list[str]) -> str:
    if cfg.data.label_col and cfg.data.label_col in df_columns:
        return str(cfg.data.label_col)
    if "__label__" in df_columns:
        return "__label__"
    if "label" in df_columns:
        return "label"
    raise RuntimeError(
        "No labels found. Provide either a label column in the CSVs via DataConfig.label_col, "
        "or configure DataConfig.label_mapping_csv (id->label)."
    )


def _save_json(path: Path, obj: Any) -> None:
    def to_jsonable(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            v = float(x)
            return v if np.isfinite(v) else None
        if isinstance(x, (np.ndarray,)):
            return to_jsonable(x.tolist())
        if isinstance(x, dict):
            return {str(k): to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_jsonable(v) for v in x]
        # Fallback: last-resort string
        return str(x)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")


def _combine_notes(*parts: Any) -> str:
    out: list[str] = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if not s:
            continue
        out.append(s)
    return "; ".join(out)


def main() -> None:
    args = parse_args()
    event_type: EventType = str(args.event_type)
    event_key = _canonical_event_type(event_type)

    cfg = get_default_config()
    np.random.seed(int(args.seed))

    out_root = (
        Path(args.output_dir)
        if args.output_dir is not None
        else (Path(cfg.output.out_dir) / "baselines" / event_key)
    )
    out_root.mkdir(parents=True, exist_ok=True)
    model_dir = out_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Resolve split dirs (supports nested Dataset/train/test)
    train_base, test_base = resolve_train_test_split_dirs(cfg.data.dataset_root)
    print(f"[train_baselines] Resolved train split dir: {str(train_base).replace('\\', '/')}")
    print(f"[train_baselines] Resolved test split dir: {str(test_base).replace('\\', '/')}")
    print(
        f"[train_baselines] Discovered train groups (folders): {[p.name for p in discover_group_folders(train_base)]}"
    )
    print(
        f"[train_baselines] Discovered test groups (folders): {[p.name for p in discover_group_folders(test_base)]}"
    )

    # Load fixed splits
    train_raw_df, test_raw_df, train_paths, test_paths = load_train_test_event_logs(
        cfg.data.dataset_root,
        event_key,
    )

    train_groups = sorted({p.parent.name for p in train_paths})
    test_groups = sorted({p.parent.name for p in test_paths})
    print(f"[train_baselines] Discovered train groups: {train_groups}")
    print(f"[train_baselines] Discovered test groups: {test_groups}")
    print(f"[train_baselines] Matched train CSV files ({len(train_paths)}):")
    for p in train_paths:
        print(f"  - {str(p).replace('\\', '/')}")
    print(f"[train_baselines] Matched test CSV files ({len(test_paths)}):")
    for p in test_paths:
        print(f"  - {str(p).replace('\\', '/')}")

    print(f"[train_baselines] Train raw dataframe shape: {train_raw_df.shape}")
    print(f"[train_baselines] Test raw dataframe shape: {test_raw_df.shape}")

    # Attach labels per split
    train_raw_df = attach_labels(
        train_raw_df,
        id_col=str(cfg.data.id_col),
        label_col=cfg.data.label_col,
        label_mapping_csv=cfg.data.resolved_label_mapping_path(),
    )
    test_raw_df = attach_labels(
        test_raw_df,
        id_col=str(cfg.data.id_col),
        label_col=cfg.data.label_col,
        label_mapping_csv=cfg.data.resolved_label_mapping_path(),
    )

    label_col = _pick_label_col(cfg, list(train_raw_df.columns))
    if label_col in train_raw_df.columns:
        print(f"[train_baselines] Raw train label distribution (including NaN), rows={len(train_raw_df)}:")
        print(train_raw_df[label_col].value_counts(dropna=False).to_string())
    if label_col in test_raw_df.columns:
        print(f"[train_baselines] Raw test label distribution (including NaN), rows={len(test_raw_df)}:")
        print(test_raw_df[label_col].value_counts(dropna=False).to_string())

    # Prepare sequences for each split (use feature_cols learned from train)
    seq_len = int(cfg.train.seq_len)
    if event_key == "email":
        X_train, y_train, feature_cols, _engineered_train = prepare_email_training_data(
            train_raw_df,
            label_col=label_col,
            user_col=str(cfg.data.email_sender_col),
            time_col=str(cfg.data.timestamp_col),
            feature_cols=None,
            seq_len=seq_len,
            debug=bool(args.debug),
        )
        X_test, y_test, _fc2, _engineered_test = prepare_email_training_data(
            test_raw_df,
            label_col=label_col,
            user_col=str(cfg.data.email_sender_col),
            time_col=str(cfg.data.timestamp_col),
            feature_cols=list(feature_cols),
            seq_len=seq_len,
            debug=False,
        )
    else:
        X_train, y_train, feature_cols, _engineered_train = prepare_login_training_data(
            train_raw_df,
            label_col=label_col,
            user_col=str(cfg.data.login_user_col),
            time_col=str(cfg.data.timestamp_col),
            feature_cols=None,
            seq_len=seq_len,
            debug=bool(args.debug),
        )
        X_test, y_test, _fc2, _engineered_test = prepare_login_training_data(
            test_raw_df,
            label_col=label_col,
            user_col=str(cfg.data.login_user_col),
            time_col=str(cfg.data.timestamp_col),
            feature_cols=list(feature_cols),
            seq_len=seq_len,
            debug=False,
        )

    if int(X_train.shape[0]) == 0:
        raise RuntimeError("No TRAIN sequences were produced. Check labels and sequence length.")
    if int(X_test.shape[0]) == 0:
        raise RuntimeError(
            "No TEST sequences were produced. Check test split contents (Dataset/train/test or Dataset/test) and sequence length."
        )

    print(f"[train_baselines] Train sequence shape: {X_train.shape}")
    print(f"[train_baselines] Test sequence shape: {X_test.shape}")

    print("[train_baselines] Model training start")

    uniq_train = np.unique(y_train)
    has_both_classes = uniq_train.size >= 2
    if not has_both_classes:
        print(
            "[train_baselines] Warning: training labels contain only one class "
            f"(unique={uniq_train.tolist()}). Skipping supervised baselines (LR/RF) and training Isolation Forest only."
        )

    threshold = float(cfg.scoring.threshold if args.threshold is None else args.threshold)

    metrics: dict[str, Any] = {}
    model_status: dict[str, str] = {}

    # Supervised baselines
    for model_name, train_fn in [
        ("logistic_regression", train_logistic_regression),
        ("random_forest", train_random_forest),
    ]:
        if not has_both_classes:
            model_status[model_name] = "skipped_single_class"
            metrics[model_name] = {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "threshold": float(threshold),
                "confusion_matrix": None,
                "classification_report": "",
                "notes": "skipped supervised baseline: only one class present in training labels",
                "metric_notes": {"roc_auc": "roc_auc unavailable: model skipped (single-class labels)"},
            }
            continue
        try:
            model = train_fn(X_train, y_train)
            y_prob = predict_supervised_probabilities(model, X_test)
            metrics[model_name] = evaluate_predictions(y_test, y_prob, threshold=threshold)
            joblib.dump(model, model_dir / f"{model_name}.joblib")
            model_status[model_name] = "ok"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            model_status[model_name] = f"error: {err}"
            metrics[model_name] = {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "threshold": float(threshold),
                "confusion_matrix": None,
                "classification_report": "",
                "notes": f"training/eval failed: {err}",
                "metric_notes": {"error": err},
            }

    # Isolation Forest (unsupervised)
    iso = train_isolation_forest(X_train)
    iso_prob = predict_isolation_forest_scores(iso, X_test)
    metrics["isolation_forest"] = evaluate_predictions(y_test, iso_prob, threshold=threshold)
    joblib.dump(iso, model_dir / "isolation_forest.joblib")
    model_status["isolation_forest"] = "ok"

    # Save outputs
    _save_json(out_root / "metrics.json", metrics)

    rows: list[dict[str, Any]] = []
    for model_name in ["logistic_regression", "random_forest", "isolation_forest"]:
        m = metrics.get(model_name, {})
        metric_notes = m.get("metric_notes") if isinstance(m, dict) else None
        roc_auc_note = ""
        if isinstance(metric_notes, dict):
            roc_auc_note = str(metric_notes.get("roc_auc", "") or "")

        rows.append(
            {
                "model": model_name,
                "status": model_status.get(model_name, "missing"),
                "threshold": m.get("threshold"),
                "accuracy": m.get("accuracy"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "roc_auc": m.get("roc_auc"),
                "roc_auc_note": roc_auc_note,
                "notes": _combine_notes(m.get("notes"), roc_auc_note),
            }
        )
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(out_root / "model_comparison.csv", index=False, na_rep="")

    report_parts: list[str] = []
    for model_name in ["logistic_regression", "random_forest", "isolation_forest"]:
        status = model_status.get(model_name, "missing")
        report = str(metrics.get(model_name, {}).get("classification_report", ""))
        report_parts.append(f"=== {model_name} (status: {status}) ===\n{report}\n")
    (out_root / "classification_reports.txt").write_text("\n".join(report_parts), encoding="utf-8")

    meta = {
        "event_type": event_key,
        "seed": int(args.seed),
        "dataset_root": str(cfg.data.dataset_root),
        "train_paths": [str(p).replace("\\", "/") for p in train_paths],
        "test_paths": [str(p).replace("\\", "/") for p in test_paths],
        "label_col": str(label_col),
        "unique_train_labels": [int(x) for x in uniq_train.astype(int).tolist()],
        "has_both_classes": bool(has_both_classes),
        "num_train_sequences": int(X_train.shape[0]),
        "num_test_sequences": int(X_test.shape[0]),
        "sequence_shape_train": list(X_train.shape),
        "sequence_shape_test": list(X_test.shape),
        "num_features": int(X_train.shape[2]),
        "feature_cols": list(feature_cols),
        "threshold": float(threshold),
        "model_status": dict(model_status),
        "config": config_to_dict(cfg),
    }
    _save_json(out_root / "meta.json", meta)

    # Console summary table
    print(f"[train_baselines] outputs: {out_root}")

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if not np.isfinite(v):
                return ""
            return f"{v:.3f}"
        return str(v)

    table_rows: list[list[str]] = []
    for _, r in comparison_df.iterrows():
        table_rows.append(
            [
                str(r.get("model", "")),
                str(r.get("status", "")),
                fmt(r.get("accuracy")),
                fmt(r.get("f1")),
                fmt(r.get("roc_auc")),
            ]
        )

    headers = ["model", "status", "acc", "f1", "roc_auc"]
    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(render_row(headers))
    print(render_row(["-" * w for w in widths]))
    for row in table_rows:
        print(render_row(row))

    print("[train_baselines] Training completion")


if __name__ == "__main__":
    main()
