from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from insider_gru.config import config_to_dict, get_default_config
from insider_gru.data import (
    attach_labels,
    discover_group_folders,
    normalize_binary_labels,
    prepare_email_training_data,
    prepare_login_training_data,
    load_train_test_event_logs,
    resolve_train_test_split_dirs,
    train_test_split_sequences,
)
from insider_gru.model import GRUAnomalyDetector, save_model_artifacts
from insider_gru.train import create_train_val_test_loaders, save_training_outputs, set_seed, train_model


EventType = Literal["email", "device", "http", "login"]


def _canonical_event_type(x: str) -> str:
    s = str(x).strip().lower()
    return "device" if s == "login" else s


def _print_label_distribution(tag: str, y: np.ndarray) -> None:
    y_arr = np.asarray(y).reshape(-1)
    if y_arr.size == 0:
        print(f"[train_gru] {tag}: empty")
        return

    uniq, counts = np.unique(y_arr, return_counts=True)
    total = float(y_arr.size)
    parts = []
    for u, c in zip(uniq.tolist(), counts.tolist(), strict=False):
        parts.append(f"{int(u)}: {int(c)} ({(100.0 * float(c) / total):.2f}%)")
    print(f"[train_gru] {tag}: " + ", ".join(parts))


def _compute_pos_weight(y_train: np.ndarray) -> float:
    y_arr = np.asarray(y_train).reshape(-1)
    pos = float(np.sum(y_arr == 1))
    neg = float(np.sum(y_arr == 0))
    if pos <= 0:
        return 1.0
    return float(neg / pos)


def _require_both_classes(tag: str, y: np.ndarray) -> None:
    y_arr = np.asarray(y).reshape(-1)
    if y_arr.size == 0:
        raise RuntimeError(f"[train_gru] {tag}: empty label array")
    uniq = np.unique(y_arr)
    if uniq.size < 2:
        _print_label_distribution(tag, y_arr)
        raise RuntimeError(f"[train_gru] {tag}: only one class present (unique={uniq.tolist()})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the GRU baseline end-to-end and save artifacts.")
    p.add_argument(
        "--event-type",
        type=str,
        choices=["email", "device", "http", "login"],
        default="email",
        help="Which event data to train on (default: email). 'login' is an alias for 'device'.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g., cpu, cuda). Defaults to cuda if available.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    event_type: EventType = str(args.event_type)
    event_key = _canonical_event_type(event_type)

    cfg = get_default_config()
    set_seed(int(args.seed))

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Outputs
    cfg.output.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.output.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.data.processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Resolve split dirs (supports nested Dataset/train/test)
    train_base, test_base = resolve_train_test_split_dirs(cfg.data.dataset_root)
    print(f"[train_gru] Resolved train split dir: {str(train_base).replace('\\', '/')}")
    print(f"[train_gru] Resolved test split dir: {str(test_base).replace('\\', '/')}")
    print(f"[train_gru] Discovered train groups (folders): {[p.name for p in discover_group_folders(train_base)]}")
    print(f"[train_gru] Discovered test groups (folders): {[p.name for p in discover_group_folders(test_base)]}")

    # Load raw CSVs from fixed dataset splits
    train_raw_df, test_raw_df, train_paths, test_paths = load_train_test_event_logs(
        cfg.data.dataset_root,
        event_key,
    )

    train_groups = sorted({p.parent.name for p in train_paths})
    test_groups = sorted({p.parent.name for p in test_paths})
    print(f"[train_gru] Discovered train groups: {train_groups}")
    print(f"[train_gru] Discovered test groups: {test_groups}")
    print(f"[train_gru] Matched train CSV files ({len(train_paths)}):")
    for p in train_paths:
        print(f"  - {str(p).replace('\\', '/')}")
    print(f"[train_gru] Matched test CSV files ({len(test_paths)}):")
    for p in test_paths:
        print(f"  - {str(p).replace('\\', '/')}")

    print(f"[train_gru] Train raw dataframe shape: {train_raw_df.shape}")
    print(f"[train_gru] Test raw dataframe shape: {test_raw_df.shape}")

    # Attach labels separately per split (keeps train/test separate).
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

    label_col: str | None = None
    if cfg.data.label_col and cfg.data.label_col in train_raw_df.columns:
        label_col = cfg.data.label_col
    elif "__label__" in train_raw_df.columns:
        label_col = "__label__"
    elif "label" in train_raw_df.columns:
        label_col = "label"

    if label_col is None:
        raise RuntimeError(
            "No labels found. Provide either a label column in the CSVs via DataConfig.label_col, "
            "or configure DataConfig.label_mapping_csv (id->label)."
        )

    print(f"[train_gru] Using label column: {label_col!r}")

    # Raw label distribution BEFORE preprocessing/cleaning.
    raw_label_series = train_raw_df[label_col] if label_col in train_raw_df.columns else None
    if raw_label_series is not None:
        print(
            f"[train_gru] Raw label distribution before preprocessing (including NaN), "
            f"dtype={raw_label_series.dtype}, rows={len(raw_label_series)}:"
        )
        print(raw_label_series.value_counts(dropna=False).to_string())

    # Label instrumentation (pre-sequences)
    raw_labels = train_raw_df[label_col]
    print(
        f"[train_gru] Raw label distribution (including NaN), dtype={raw_labels.dtype}, rows={len(raw_labels)}:"
    )
    print(raw_labels.value_counts(dropna=False).to_string())

    norm_labels = normalize_binary_labels(raw_labels)
    print("[train_gru] Normalized/binarized labels (pre-prepare_*):")
    print(norm_labels.value_counts(dropna=False).to_string())

    test_norm_labels = normalize_binary_labels(test_raw_df[label_col]) if label_col in test_raw_df.columns else None
    if test_norm_labels is not None:
        print("[train_gru] Normalized/binarized labels (test split, pre-prepare_*):")
        print(test_norm_labels.value_counts(dropna=False).to_string())

    # If labels already collapsed to one class here, training is unlikely to be meaningful.
    _require_both_classes("Labels after binary normalization (pre-prepare_*)", norm_labels.to_numpy())

    # Prepare sequences (train split)
    if event_key == "email":
        X_train_all, y_train_all, feature_cols, _df_train = prepare_email_training_data(
            train_raw_df,
            label_col=label_col,
            user_col=str(cfg.data.email_sender_col),
            time_col=str(cfg.data.timestamp_col),
            feature_cols=None,
            seq_len=int(cfg.gru.sequence_length),
            debug=True,
        )
    else:
        X_train_all, y_train_all, feature_cols, _df_train = prepare_login_training_data(
            train_raw_df,
            label_col=label_col,
            user_col=str(cfg.data.login_user_col),
            time_col=str(cfg.data.timestamp_col),
            feature_cols=None,
            seq_len=int(cfg.gru.sequence_length),
            debug=True,
        )

    # Prepare sequences (test split) using the feature columns learned from train.
    if event_key == "email":
        X_test, y_test, _fc2, _df_test = prepare_email_training_data(
            test_raw_df,
            label_col=label_col,
            user_col=str(cfg.data.email_sender_col),
            time_col=str(cfg.data.timestamp_col),
            feature_cols=list(feature_cols),
            seq_len=int(cfg.gru.sequence_length),
            debug=False,
        )
    else:
        X_test, y_test, _fc2, _df_test = prepare_login_training_data(
            test_raw_df,
            label_col=label_col,
            user_col=str(cfg.data.login_user_col),
            time_col=str(cfg.data.timestamp_col),
            feature_cols=list(feature_cols),
            seq_len=int(cfg.gru.sequence_length),
            debug=False,
        )

    if X_train_all.shape[0] == 0:
        raise RuntimeError("No sequences were produced. Check label availability and sequence length.")

    print(f"[train_gru] Train sequence shape: {X_train_all.shape}")
    print(f"[train_gru] Test sequence shape: {X_test.shape}")

    uniq, counts = np.unique(y_train_all, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist(), strict=False)}
    print(f"[train_gru] Labels after sequence building: {dist}")
    _print_label_distribution("Class distribution (train split)", y_train_all)
    _require_both_classes("Train split labels after sequence building", y_train_all)

    # Split train further into train/val so we can tune the decision threshold on validation.
    val_split = float(getattr(cfg, "train").val_split) if hasattr(cfg, "train") else 0.2
    X_tr, X_val, y_tr, y_val = train_test_split_sequences(
        X_train_all,
        y_train_all,
        test_size=val_split,
        random_state=int(args.seed),
        stratify=True,
    )

    _print_label_distribution("Class distribution after split (train)", y_tr)
    _print_label_distribution("Class distribution after split (val)", y_val)
    _print_label_distribution("Class distribution (test split)", y_test)

    _require_both_classes("Training split labels", y_tr)
    if np.unique(y_val).size < 2:
        print("[train_gru] WARNING: validation split has only one class; threshold tuning may be uninformative.")
    if np.unique(y_test).size < 2:
        print("[train_gru] WARNING: test split has only one class; reported metrics may be misleading.")

    pos_weight = _compute_pos_weight(y_tr)
    print(f"[train_gru] Computed pos_weight from y_train: {pos_weight:.4f}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_test,
        y_test,
        batch_size=int(cfg.gru.batch_size),
    )

    # Instantiate and train model
    model = GRUAnomalyDetector(
        input_size=int(X_tr.shape[2]),
        hidden_size=int(cfg.gru.hidden_dim),
        num_layers=int(cfg.gru.num_layers),
        dropout=float(cfg.gru.dropout),
    )

    model, history, final_eval = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=int(cfg.gru.num_epochs),
        learning_rate=float(cfg.gru.learning_rate),
        device=str(device),
        val_loader=val_loader,
        pos_weight=float(pos_weight),
        tune_threshold=True,
        threshold_grid=np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60], dtype=float),
    )

    # Save artifacts
    extra_metadata = {
        "event_type": event_key,
        "device": str(device),
        "seed": int(args.seed),
        "dataset_root": str(cfg.data.dataset_root),
        "train_paths": [str(p).replace("\\", "/") for p in train_paths],
        "test_paths": [str(p).replace("\\", "/") for p in test_paths],
        "num_train_sequences": int(X_train_all.shape[0]),
        "num_test_sequences": int(X_test.shape[0]),
        "train_sequence_shape": list(X_train_all.shape),
        "test_sequence_shape": list(X_test.shape),
        "num_features": int(X_train_all.shape[2]),
        "val_split": float(val_split),
        "pos_weight": float(pos_weight),
        "config": config_to_dict(cfg),
    }

    save_model_artifacts(
        model,
        cfg.output.model_dir,
        metadata={"feature_cols": feature_cols, **extra_metadata},
        model_config={
            "input_size": int(X_tr.shape[2]),
            "hidden_size": int(cfg.gru.hidden_dim),
            "num_layers": int(cfg.gru.num_layers),
            "dropout": float(cfg.gru.dropout),
        },
    )

    save_training_outputs(
        cfg.output.reports_dir,
        history=history,
        final_eval=final_eval,
        feature_cols=feature_cols,
        extra_metadata=extra_metadata,
    )

    print("[train_gru] Training completion")

    print(
        "[train_gru] Evaluation results: "
        f"accuracy={final_eval.get('accuracy')}, f1={final_eval.get('f1')}, "
        f"precision={final_eval.get('precision')}, recall={final_eval.get('recall')}, "
        f"roc_auc={final_eval.get('roc_auc')}, threshold={final_eval.get('selected_threshold', final_eval.get('threshold'))}"
    )

    # Print summary
    summary = {
        "num_train_sequences": int(X_train_all.shape[0]),
        "num_test_sequences": int(X_test.shape[0]),
        "train_sequence_shape": list(X_train_all.shape),
        "test_sequence_shape": list(X_test.shape),
        "num_features": int(X_train_all.shape[2]),
        "final_accuracy": final_eval.get("accuracy"),
        "final_precision": final_eval.get("precision"),
        "final_recall": final_eval.get("recall"),
        "final_f1": final_eval.get("f1"),
        "final_roc_auc": final_eval.get("roc_auc"),
        "best_threshold": final_eval.get("best_threshold"),
        "pos_weight": final_eval.get("pos_weight"),
        "outputs": {
            "model_dir": str(cfg.output.model_dir),
            "reports_dir": str(cfg.output.reports_dir),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
