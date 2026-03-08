from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, overload

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from insider_gru.model import GRUClassifier, save_model_artifacts


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test DataLoaders for fixed-length sequence classification.

    Inputs are expected as NumPy arrays:
    - `X_*`: (N, T, F)
    - `y_*`: (N,) or (N, 1)

    The returned batches are `(X, y)` tensors.
    """

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(np.asarray(y_train).reshape(-1), dtype=torch.float32)
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32)
    y_test_t = torch.as_tensor(np.asarray(y_test).reshape(-1), dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def create_train_val_test_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for fixed-length sequence classification."""

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(np.asarray(y_train).reshape(-1), dtype=torch.float32)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(np.asarray(y_val).reshape(-1), dtype=torch.float32)
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32)
    y_test_t = torch.as_tensor(np.asarray(y_test).reshape(-1), dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Support loaders yielding (X, y) or (X, y, lengths)."""

    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Expected batch tuple/list, got {type(batch).__name__}")
    if len(batch) == 2:
        X, y = batch
        return X, y, None
    if len(batch) == 3:
        X, y, lengths = batch
        return X, y, lengths
    raise ValueError(f"Expected batch of length 2 or 3; got {len(batch)}")


def _bce_logits_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits_1d = logits.view(-1)
    targets_1d = targets.view(-1)
    return criterion(logits_1d, targets_1d)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> dict[str, float]:
    """Train for one epoch.

    Returns a dict containing average loss and average predicted probability.
    """

    dev = torch.device(device)
    model.to(dev)
    model.train()

    losses: list[float] = []
    prob_means: list[float] = []

    for batch in loader:
        X, y, lengths = _unpack_batch(batch)
        X = X.to(dev)
        y = y.to(dev)

        optimizer.zero_grad(set_to_none=True)
        if lengths is None:
            logits = model(X)
        else:
            logits = model(X, lengths.to(dev))
        loss = _bce_logits_loss(criterion, logits, y)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        prob_means.append(float(torch.sigmoid(logits.detach()).mean().cpu().item()))

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "prob_mean": float(np.mean(prob_means)) if prob_means else 0.0,
    }


def _evaluate_model_fixed(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a model on a loader.

    Collects raw arrays (y_true, y_prob, y_pred) and computes loss + common
    binary classification metrics.
    """

    dev = torch.device(device)
    model.to(dev)
    model.eval()

    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            X, y, lengths = _unpack_batch(batch)
            X = X.to(dev)
            y = y.to(dev)

            if lengths is None:
                logits = model(X)
            else:
                logits = model(X, lengths.to(dev))

            loss = _bce_logits_loss(criterion, logits, y)
            losses.append(float(loss.detach().cpu().item()))

            probs = torch.sigmoid(logits.detach()).view(-1).cpu().numpy()
            y_true.append(y.detach().view(-1).cpu().numpy())
            y_prob.append(probs)

    y_true_arr = np.concatenate(y_true, axis=0) if y_true else np.zeros((0,), dtype=np.float32)
    y_prob_arr = np.concatenate(y_prob, axis=0) if y_prob else np.zeros((0,), dtype=np.float32)
    y_pred_arr = (y_prob_arr >= float(threshold)).astype(int)

    out: dict[str, Any] = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "threshold": float(threshold),
        "y_true": y_true_arr,
        "y_prob": y_prob_arr,
        "y_pred": y_pred_arr,
    }

    if y_true_arr.size == 0:
        out.update({
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "classification_report": "",
        })
        return out

    out["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
    out["precision"] = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    out["recall"] = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    out["f1"] = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    try:
        out["roc_auc"] = (
            float(roc_auc_score(y_true_arr, y_prob_arr)) if np.unique(y_true_arr).size > 1 else None
        )
    except Exception:
        out["roc_auc"] = None

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()
    out["classification_report"] = classification_report(y_true_arr, y_pred_arr, zero_division=0)
    return out


def _select_threshold_max_f1(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Select a probability threshold by maximizing F1 over a grid."""

    y_true_arr = np.asarray(y_true).reshape(-1)
    y_prob_arr = np.asarray(y_prob).reshape(-1)

    if y_true_arr.size == 0 or np.unique(y_true_arr).size < 2:
        return 0.5

    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    # Compute F1 at each threshold. On ties, prefer thresholds closest to 0.5.
    rows: list[tuple[float, float]] = []
    for t in grid:
        tt = float(t)
        y_pred = (y_prob_arr >= tt).astype(int)
        rows.append((tt, float(f1_score(y_true_arr, y_pred, zero_division=0))))

    best_f1 = max((f for _, f in rows), default=-1.0)
    best_candidates = [t for t, f in rows if f == best_f1]
    if not best_candidates:
        return 0.5
    best_candidates.sort(key=lambda t: (abs(t - 0.5), t))
    return float(best_candidates[0])


def _threshold_metrics_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float],
    *,
    split: str,
) -> pd.DataFrame:
    """Compute common metrics at multiple thresholds.

    This is intended for threshold tuning/analysis; ROC-AUC is independent of the threshold
    and is repeated per-row for convenience.
    """

    y_true_arr = np.asarray(y_true).reshape(-1).astype(int)
    y_prob_arr = np.asarray(y_prob).reshape(-1).astype(float)

    if y_true_arr.size != y_prob_arr.size:
        raise ValueError(
            f"y_true and y_prob must have same length; got {y_true_arr.size} vs {y_prob_arr.size}"
        )

    roc_auc: float | None
    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr)) if np.unique(y_true_arr).size > 1 else None
    except Exception:
        roc_auc = None

    rows: list[dict[str, Any]] = []
    for t in thresholds:
        tt = float(t)
        y_pred = (y_prob_arr >= tt).astype(int)
        rows.append(
            {
                "split": str(split),
                "threshold": tt,
                "accuracy": float(accuracy_score(y_true_arr, y_pred)) if y_true_arr.size else 0.0,
                "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)) if y_true_arr.size else 0.0,
                "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)) if y_true_arr.size else 0.0,
                "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)) if y_true_arr.size else 0.0,
                "roc_auc": roc_auc,
            }
        )

    return pd.DataFrame(rows)


def _train_model_fixed(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str | torch.device = "cpu",
    *,
    val_loader: DataLoader | None = None,
    pos_weight: float | None = None,
    tune_threshold: bool = True,
    threshold_grid: np.ndarray | None = None,
    progress: bool = True,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    """Train a binary classifier with BCEWithLogitsLoss.

    Returns: (trained_model, history, final_eval)
    - history: list of dicts (one per epoch)
    - final_eval: evaluation dict produced by `evaluate_model`
    """

    dev = torch.device(device)
    model.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    inferred_pos_weight: float | None = None
    if pos_weight is None:
        # Infer from the training labels when train_loader uses a TensorDataset.
        try:
            ds = train_loader.dataset
            if isinstance(ds, TensorDataset) and len(ds.tensors) >= 2:
                y_np = ds.tensors[1].detach().cpu().numpy().reshape(-1)
                pos = float(np.sum(y_np == 1))
                neg = float(np.sum(y_np == 0))
                inferred_pos_weight = float(neg / pos) if pos > 0 else 1.0
        except Exception:
            inferred_pos_weight = None

    effective_pos_weight = float(pos_weight) if pos_weight is not None else inferred_pos_weight
    if effective_pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(effective_pos_weight, device=dev))
    else:
        criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, Any]] = []
    final_eval: dict[str, Any] = {}

    eval_loader = val_loader if val_loader is not None else test_loader

    if progress:
        try:
            train_batches = int(len(train_loader))
        except Exception:
            train_batches = -1
        try:
            val_batches = int(len(eval_loader))
        except Exception:
            val_batches = -1

        print(
            "[train] Model training start: "
            f"epochs={int(num_epochs)}, lr={float(learning_rate)}, device={str(dev)}, "
            f"train_batches={train_batches}, val_batches={val_batches}, "
            f"tune_threshold={bool(tune_threshold and (val_loader is not None))}"
        )

    for epoch in range(1, int(num_epochs) + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, dev)
        ev = _evaluate_model_fixed(model, eval_loader, criterion, dev, threshold=0.5)

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_prob_mean": tr.get("prob_mean"),
                "val_loss": ev["loss"],
                "val_accuracy": ev.get("accuracy"),
                "val_precision": ev.get("precision"),
                "val_recall": ev.get("recall"),
                "val_f1": ev.get("f1"),
                "val_roc_auc": ev.get("roc_auc"),
            }
        )

        if progress:
            msg_parts = [
                f"epoch {epoch}/{int(num_epochs)}",
                f"train_loss={tr['loss']:.6f}",
                f"val_loss={float(ev.get('loss', 0.0)):.6f}",
            ]

            if ev.get("accuracy") is not None:
                msg_parts.append(f"val_acc={float(ev['accuracy']):.4f}")
            if ev.get("precision") is not None:
                msg_parts.append(f"val_prec={float(ev['precision']):.4f}")
            if ev.get("recall") is not None:
                msg_parts.append(f"val_rec={float(ev['recall']):.4f}")
            if ev.get("f1") is not None:
                msg_parts.append(f"val_f1={float(ev['f1']):.4f}")
            if ev.get("roc_auc") is not None:
                msg_parts.append(f"val_roc_auc={float(ev['roc_auc']):.4f}")

            print("[train] " + ", ".join(msg_parts))

    # Threshold tuning using the validation set (if available), then evaluate on the held-out test set.
    tuned_threshold = 0.5
    thresholds = (
        np.asarray(threshold_grid, dtype=float).reshape(-1)
        if threshold_grid is not None
        else np.linspace(0.05, 0.95, 19)
    )

    threshold_comparison_df: pd.DataFrame | None = None

    if tune_threshold and val_loader is not None:
        val_ev = _evaluate_model_fixed(model, val_loader, criterion, dev, threshold=0.5)
        tuned_threshold = _select_threshold_max_f1(val_ev["y_true"], val_ev["y_prob"], grid=thresholds)

        if progress:
            # Report best threshold selection at a high level.
            print(f"[train] Threshold tuning complete: best_threshold={float(tuned_threshold):.4f}")

        # Save a small comparison table for analysis (val + test).
        test_ev_for_table = _evaluate_model_fixed(model, test_loader, criterion, dev, threshold=0.5)
        val_tbl = _threshold_metrics_table(val_ev["y_true"], val_ev["y_prob"], thresholds, split="val")
        test_tbl = _threshold_metrics_table(
            test_ev_for_table["y_true"], test_ev_for_table["y_prob"], thresholds, split="test"
        )
        threshold_comparison_df = pd.concat([val_tbl, test_tbl], ignore_index=True)

    final_eval = _evaluate_model_fixed(model, test_loader, criterion, dev, threshold=tuned_threshold)
    final_eval["best_threshold"] = float(tuned_threshold)
    final_eval["selected_threshold"] = float(tuned_threshold)
    final_eval["thresholds_evaluated"] = [float(t) for t in thresholds.tolist()]
    if threshold_comparison_df is not None and not threshold_comparison_df.empty:
        final_eval["threshold_comparison"] = threshold_comparison_df.to_dict(orient="records")
    if effective_pos_weight is not None:
        final_eval["pos_weight"] = float(effective_pos_weight)

    if progress:
        acc = final_eval.get("accuracy")
        f1 = final_eval.get("f1")
        prec = final_eval.get("precision")
        rec = final_eval.get("recall")
        roc = final_eval.get("roc_auc")
        thr = final_eval.get("selected_threshold", final_eval.get("threshold"))
        print(
            "[train] Evaluation results: "
            f"threshold={thr}, accuracy={acc}, precision={prec}, recall={rec}, f1={f1}, roc_auc={roc}"
        )
        print("[train] Training completion")

    return model, history, final_eval


def evaluate_model(
    model: nn.Module,
    loader: DataLoader | None = None,
    criterion: nn.Module | None = None,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
    **legacy_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate model using either the new or legacy API.

    New API (fixed-length, recommended):
        `evaluate_model(model, loader, criterion, device, threshold=0.5)`

    Legacy API (length-aware, kept for compatibility):
        `evaluate_model(model, X=..., y=..., lengths=..., batch_size=..., threshold=..., device=...)`
    """

    if legacy_kwargs:
        return evaluate_model_legacy(model, device=str(device), threshold=threshold, **legacy_kwargs)

    if loader is None or criterion is None:
        raise TypeError("evaluate_model requires (loader, criterion) for the new API")
    return _evaluate_model_fixed(model, loader, criterion, device, threshold=threshold)


def train_model(
    model: nn.Module | None = None,
    train_loader: DataLoader | None = None,
    test_loader: DataLoader | None = None,
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    device: str | torch.device = "cpu",
    *,
    val_loader: DataLoader | None = None,
    pos_weight: float | None = None,
    tune_threshold: bool = True,
    threshold_grid: np.ndarray | None = None,
    progress: bool = True,
    **legacy_kwargs: Any,
) -> Any:
    """Train model using either the new or legacy API.

    New API (fixed-length, recommended):
        `train_model(model, train_loader, test_loader, num_epochs, learning_rate, device="cpu")`

    Legacy API (length-aware, kept for compatibility):
        `train_model(X=..., y=..., lengths=..., input_size=..., hidden_size=..., ...)`
    """

    if legacy_kwargs:
        # Preserve the original return type: (GRUClassifier, TrainHistory, summary)
        return train_model_legacy(**legacy_kwargs)

    if model is None or train_loader is None or test_loader is None or num_epochs is None or learning_rate is None:
        raise TypeError(
            "train_model requires (model, train_loader, test_loader, num_epochs, learning_rate) for the new API"
        )
    return _train_model_fixed(
        model,
        train_loader,
        test_loader,
        num_epochs=int(num_epochs),
        learning_rate=float(learning_rate),
        device=device,
        val_loader=val_loader,
        pos_weight=pos_weight,
        tune_threshold=bool(tune_threshold),
        threshold_grid=threshold_grid,
        progress=bool(progress),
    )


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_training_outputs(
    output_dir: str | Path,
    history: list[dict[str, Any]],
    final_eval: dict[str, Any],
    feature_cols: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Persist training outputs in a Streamlit/report-friendly format."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) training_history.csv
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)

    # 2) metrics.json (exclude big arrays; we store predictions separately)
    metrics = {
        k: v
        for k, v in final_eval.items()
        if k not in {"y_true", "y_prob", "y_pred", "threshold_comparison"}
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(_json_sanitize(metrics), indent=2),
        encoding="utf-8",
    )

    # 2b) threshold_comparison.csv (if present)
    if "threshold_comparison" in final_eval and final_eval.get("threshold_comparison") is not None:
        try:
            tc = final_eval.get("threshold_comparison")
            tc_df = pd.DataFrame(tc)
            if not tc_df.empty:
                tc_df.to_csv(out_dir / "threshold_comparison.csv", index=False)
        except Exception:
            pass

    # 3) classification_report.txt
    report = final_eval.get("classification_report") or ""
    (out_dir / "classification_report.txt").write_text(str(report), encoding="utf-8")

    # 4) predictions.csv if available
    if all(k in final_eval for k in ("y_true", "y_prob", "y_pred")):
        try:
            pred_df = pd.DataFrame(
                {
                    "y_true": np.asarray(final_eval["y_true"]).reshape(-1),
                    "y_prob": np.asarray(final_eval["y_prob"]).reshape(-1),
                    "y_pred": np.asarray(final_eval["y_pred"]).reshape(-1),
                }
            )
            pred_df.to_csv(out_dir / "predictions.csv", index=False)
        except Exception:
            # Be permissive: if shapes are unexpected, skip predictions.
            pass

    # 5) metadata.json
    metadata: dict[str, Any] = {}
    if feature_cols is not None:
        metadata["feature_cols"] = list(feature_cols)
    if extra_metadata:
        metadata.update(extra_metadata)
    (out_dir / "metadata.json").write_text(
        json.dumps(_json_sanitize(metadata), indent=2),
        encoding="utf-8",
    )

    # NOTE:
    # The public fixed-length APIs `train_model(...)` and `evaluate_model(...)` are defined earlier
    # in this module. Legacy (length-aware) helpers and `train_model_legacy` remain below.


@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]
    train_acc: list[float]
    val_acc: list[float]


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, lengths: np.ndarray):
        self.X = X
        self.y = y
        self.lengths = lengths

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.lengths[idx]


def _collate(batch):
    X, y, L = zip(*batch)
    X_t = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y), dtype=torch.float32)
    L_t = torch.tensor(np.asarray(L), dtype=torch.int64)
    return X_t, y_t, L_t


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    val_split: float,
    seed: int,
):
    if X.shape[0] == 0:
        return (X, y, lengths), (X, y, lengths)

    stratify = y if len(np.unique(y)) > 1 else None
    train_idx, val_idx = train_test_split(
        np.arange(X.shape[0]),
        test_size=val_split,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return (X[train_idx], y[train_idx], lengths[train_idx]), (X[val_idx], y[val_idx], lengths[val_idx])


def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    # pos_weight for BCEWithLogitsLoss
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos)


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for X, _, L in loader:
        X = X.to(device)
        L = L.to(device)
        logits = model(X, L)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)


def select_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    # Maximize F1 on a coarse grid.
    if len(y_true) == 0:
        return 0.5
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        preds = (probs >= t).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def train_model_legacy(
    *,
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    early_stopping_patience: int,
    val_split: float,
    seed: int,
    device: str | None = None,
) -> tuple[GRUClassifier, TrainHistory, dict[str, Any]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    (X_tr, y_tr, L_tr), (X_va, y_va, L_va) = split_train_val(X, y, lengths, val_split, seed)

    train_ds = SequenceDataset(X_tr, y_tr, L_tr)
    val_ds = SequenceDataset(X_va, y_va, L_va)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    model = GRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(dev)

    pos_weight = compute_pos_weight(y_tr).to(dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = TrainHistory(train_loss=[], val_loss=[], train_acc=[], val_acc=[])

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_losses: list[float] = []
        tr_true: list[float] = []
        tr_pred: list[float] = []

        for Xb, yb, Lb in tqdm(train_loader, desc=f"epoch {epoch}/{max_epochs}", leave=False):
            Xb = Xb.to(dev)
            yb = yb.to(dev)
            Lb = Lb.to(dev)

            optim.zero_grad(set_to_none=True)
            logits = model(Xb, Lb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

            tr_losses.append(float(loss.detach().cpu().item()))
            tr_true.append(yb.detach().cpu().numpy())
            tr_pred.append(torch.sigmoid(logits).detach().cpu().numpy())

        tr_true_arr = np.concatenate(tr_true)
        tr_pred_arr = np.concatenate(tr_pred)
        tr_acc = accuracy_score(tr_true_arr, (tr_pred_arr >= 0.5).astype(int))

        model.eval()
        va_losses: list[float] = []
        va_true: list[float] = []
        va_pred: list[float] = []
        with torch.no_grad():
            for Xb, yb, Lb in val_loader:
                Xb = Xb.to(dev)
                yb = yb.to(dev)
                Lb = Lb.to(dev)
                logits = model(Xb, Lb)
                loss = criterion(logits, yb)
                va_losses.append(float(loss.detach().cpu().item()))
                va_true.append(yb.detach().cpu().numpy())
                va_pred.append(torch.sigmoid(logits).detach().cpu().numpy())

        va_true_arr = np.concatenate(va_true) if va_true else np.zeros((0,))
        va_pred_arr = np.concatenate(va_pred) if va_pred else np.zeros((0,))
        va_acc = accuracy_score(va_true_arr, (va_pred_arr >= 0.5).astype(int)) if len(va_true_arr) else 0.0

        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_loss = float(np.mean(va_losses)) if va_losses else 0.0

        history.train_loss.append(tr_loss)
        history.val_loss.append(va_loss)
        history.train_acc.append(float(tr_acc))
        history.val_acc.append(float(va_acc))

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Choose threshold on validation
    val_probs = predict_proba(model, val_loader, dev)
    best_threshold = select_threshold(y_va, val_probs)

    summary = {
        "device": str(dev),
        "best_val_loss": best_val,
        "epochs_ran": len(history.train_loss),
        "pos_weight": float(pos_weight.detach().cpu().item()),
        "best_threshold": float(best_threshold),
    }
    return model, history, summary


def evaluate_model_legacy(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    threshold: float,
    device: str | None = None,
) -> dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = model.to(dev)

    ds = SequenceDataset(X, y, lengths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    probs = predict_proba(model, loader, dev)
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y, preds) if len(y) else 0.0
    pr, rc, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y, preds, labels=[0, 1])
    report = classification_report(y, preds, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "roc_auc": float(auc) if not (auc != auc) else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_artifacts(
    *,
    model: nn.Module,
    preprocessor,
    config: dict[str, Any],
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preproc_path = out_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preproc_path)

    model_path, meta_path = save_model_artifacts(
        model,
        out_dir,
        metadata={"config": _json_sanitize(config)},
    )

    return {
        "model": str(model_path),
        "preprocessor": str(preproc_path),
        "meta": str(meta_path),
    }


def load_artifacts(
    *,
    model_class: type[GRUClassifier],
    model_kwargs: dict[str, Any],
    model_dir: Path,
):
    model_path = model_dir / "model.pt"
    preproc_path = model_dir / "preprocessor.joblib"

    if not model_path.exists() or not preproc_path.exists():
        raise FileNotFoundError(f"Missing model artifacts in {model_dir}")

    preprocessor = joblib.load(preproc_path)
    model = model_class(**model_kwargs)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, preprocessor
