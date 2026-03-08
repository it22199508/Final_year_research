from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from insider_gru.config import OutputConfig, TrainConfig
from insider_gru.data import (
    PreparedData,
    attach_labels,
    build_preprocessor,
    create_sequences,
    load_event_csvs,
    normalize_binary_labels,
    prepare_dataframe,
)
from insider_gru.plots import plot_confusion_matrix, plot_history
from insider_gru.train import evaluate_model, save_artifacts, train_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train a GRU-based device/login detector on the CERT device logs using the existing "
            "PyTorch training pipeline (no TensorFlow required)."
        )
    )
    p.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    p.add_argument(
        "--train-glob",
        type=str,
        default="train/test/R*/device-train-data.csv",
        help="Glob under dataset-root for the TRAIN split",
    )
    p.add_argument(
        "--test-glob",
        type=str,
        default="train/R*/device-test-data*.csv",
        help="Glob under dataset-root for the TEST split",
    )
    p.add_argument(
        "--label-mapping",
        type=Path,
        default=None,
        help=(
            "Optional id->label CSV with columns: id,label. "
            "If not provided, the script expects a label column in the event CSVs."
        ),
    )
    p.add_argument("--feature-mode", type=str, choices=["minimal", "all"], default="all")

    p.add_argument("--seq-len", type=int, default=40)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--label-strategy", type=str, choices=["any_positive", "last"], default="last")
    p.add_argument("--window-anchor", type=str, choices=["start", "end"], default="end")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/login"),
        help="Output directory (default: outputs/login)",
    )
    return p.parse_args()


def write_training_history(history, out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(asdict(history), indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for i in range(len(history.train_loss)):
            w.writerow(
                [
                    i + 1,
                    history.train_loss[i],
                    history.val_loss[i],
                    history.train_acc[i],
                    history.val_acc[i],
                ]
            )


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))

    train_cfg = TrainConfig(
        seq_len=int(args.seq_len),
        stride=int(args.stride),
        label_strategy=str(args.label_strategy),
        window_anchor=str(args.window_anchor),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        early_stopping_patience=int(args.patience),
        val_split=float(args.val_split),
        seed=int(args.seed),
        threshold=float(args.threshold),
    )

    out_cfg = OutputConfig(
        out_dir=args.out_dir,
        model_dir=args.out_dir / "model",
        plots_dir=args.out_dir / "plots",
        reports_dir=args.out_dir / "reports",
    )

    root = Path(args.dataset_root)
    train_paths = sorted(root.glob(str(args.train_glob)))
    test_paths = sorted(root.glob(str(args.test_glob)))
    if not train_paths:
        raise FileNotFoundError(f"No train files matched: {root}/{args.train_glob}")
    if not test_paths:
        raise FileNotFoundError(f"No test files matched: {root}/{args.test_glob}")

    train_df = load_event_csvs(train_paths)
    test_df = load_event_csvs(test_paths)

    # Device logs are schema: id,date,user,pc,activity (no label col by default)
    train_df = attach_labels(train_df, id_col="id", label_col=None, label_mapping_csv=args.label_mapping)
    test_df = attach_labels(test_df, id_col="id", label_col=None, label_mapping_csv=args.label_mapping)

    label_col = None
    if "label" in train_df.columns:
        label_col = "label"
    elif "__label__" in train_df.columns:
        label_col = "__label__"

    if label_col is None:
        raise RuntimeError(
            "No labels found for device/login data. Provide either: "
            "(1) a label column named 'label' in the CSVs, or "
            "(2) --label-mapping <csv> with columns: id,label. "
            "Tip: create a template via scripts/make_label_mapping_template.py using device globs."
        )

    prep_train: PreparedData = prepare_dataframe(
        train_df,
        timestamp_col="date",
        id_col="id",
        group_col_candidates=("user",),
        label_col=label_col,
        feature_mode=str(args.feature_mode),
    )
    prep_test: PreparedData = prepare_dataframe(
        test_df,
        timestamp_col="date",
        id_col="id",
        group_col_candidates=("user",),
        label_col=label_col,
        feature_mode=str(args.feature_mode),
    )

    y_train_evt = normalize_binary_labels(prep_train.df[label_col]).to_numpy()
    y_test_evt = normalize_binary_labels(prep_test.df[label_col]).to_numpy()

    preprocessor = build_preprocessor(prep_train.df, prep_train.feature_cols)
    X_train_evt = preprocessor.fit_transform(prep_train.df[prep_train.feature_cols])
    X_test_evt = preprocessor.transform(prep_test.df[prep_test.feature_cols])

    train_seq = create_sequences(
        prep_train.df,
        group_col=prep_train.group_col,
        timestamp_col=prep_train.timestamp_col,
        feature_matrix=X_train_evt,
        label=y_train_evt,
        seq_len=train_cfg.seq_len,
        stride=train_cfg.stride,
        label_strategy=train_cfg.label_strategy,
        anchor=train_cfg.window_anchor,
    )
    test_seq = create_sequences(
        prep_test.df,
        group_col=prep_test.group_col,
        timestamp_col=prep_test.timestamp_col,
        feature_matrix=X_test_evt,
        label=y_test_evt,
        seq_len=train_cfg.seq_len,
        stride=train_cfg.stride,
        label_strategy=train_cfg.label_strategy,
        anchor=train_cfg.window_anchor,
    )

    if train_seq.X.shape[0] == 0:
        raise RuntimeError(
            "No sequences created from device/login training data. "
            "Try reducing --seq-len or check timestamps/grouping."
        )

    unique_train_labels = np.unique(train_seq.y)
    if unique_train_labels.size < 2:
        raise RuntimeError(
            "Training labels contain only one class after sequence building "
            f"(unique={unique_train_labels.tolist()}). "
            "Update your label mapping so you have both 0 and 1 labels."
        )

    model, history, summary = train_model(
        X=train_seq.X,
        y=train_seq.y,
        lengths=train_seq.lengths,
        input_size=train_seq.X.shape[2],
        hidden_size=train_cfg.hidden_size,
        num_layers=train_cfg.num_layers,
        dropout=train_cfg.dropout,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        batch_size=train_cfg.batch_size,
        max_epochs=train_cfg.max_epochs,
        early_stopping_patience=train_cfg.early_stopping_patience,
        val_split=train_cfg.val_split,
        seed=train_cfg.seed,
    )

    metrics = evaluate_model(
        model,
        X=test_seq.X,
        y=test_seq.y,
        lengths=test_seq.lengths,
        batch_size=train_cfg.batch_size,
        threshold=train_cfg.threshold,
    )

    out_cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_cfg.model_dir.mkdir(parents=True, exist_ok=True)
    out_cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    out_cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    plot_history(history, out_cfg.plots_dir / "history.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        labels=["0", "1"],
        output_path=out_cfg.plots_dir / "confusion_matrix.png",
    )

    write_training_history(
        history,
        out_csv=out_cfg.reports_dir / "training_history.csv",
        out_json=out_cfg.reports_dir / "training_history.json",
    )

    (out_cfg.reports_dir / "metrics.json").write_text(json.dumps({**summary, **metrics}, indent=2), encoding="utf-8")
    (out_cfg.reports_dir / "classification_report.txt").write_text(metrics["classification_report"], encoding="utf-8")

    artifacts = save_artifacts(
        model=model,
        preprocessor=preprocessor,
        config={
            "data": {
                "dataset_root": str(root),
                "train_glob": str(args.train_glob),
                "test_glob": str(args.test_glob),
                "timestamp_col": "date",
                "id_col": "id",
                "label_mapping_csv": str(args.label_mapping) if args.label_mapping else None,
                "group_col_candidates": ["user"],
                "feature_mode": str(args.feature_mode),
            },
            "train": train_cfg.__dict__,
        },
        out_dir=out_cfg.model_dir,
    )

    print("Training summary:")
    print(json.dumps(summary, indent=2))
    print("Test metrics:")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print("Saved:")
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
