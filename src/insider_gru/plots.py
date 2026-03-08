from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def _ensure_path(p: str | Path) -> Path:
    path = p if isinstance(p, Path) else Path(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]] | np.ndarray,
    labels: Sequence[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a simple confusion matrix figure.

    Args:
        cm: Confusion matrix shaped (n_classes, n_classes).
        labels: Class labels for axis tick labels.
        output_path: Where to save the PNG.
        title: Plot title.
    """

    out = _ensure_path(output_path)
    cm_arr = np.asarray(cm)

    if cm_arr.ndim != 2 or cm_arr.shape[0] != cm_arr.shape[1]:
        raise ValueError(f"Expected square confusion matrix; got shape={cm_arr.shape}")

    n = int(cm_arr.shape[0])
    if len(labels) != n:
        raise ValueError(f"labels length must equal cm size ({n}); got {len(labels)}")

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm_arr, interpolation="nearest", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=list(labels),
        yticklabels=list(labels),
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = float(cm_arr.max()) / 2.0 if cm_arr.size else 0.0
    for i in range(n):
        for j in range(n):
            v = int(cm_arr[i, j])
            ax.text(
                j,
                i,
                f"{v}",
                ha="center",
                va="center",
                color="white" if v > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    output_path: str | Path,
    title: str = "ROC Curve",
) -> None:
    """Plot and save ROC curve with AUC.

    Computes FPR/TPR via sklearn, computes AUC, and saves a PNG.

    Notes:
        If only one class is present in y_true, AUC is undefined; in that case the
        function will still plot the diagonal baseline and annotate AUC as N/A.
    """

    out = _ensure_path(output_path)
    y_t = np.asarray(y_true).reshape(-1).astype(int)
    y_p = np.asarray(y_prob).reshape(-1).astype(float)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="chance")

    auc_value: float | None = None
    if y_t.size > 0 and np.unique(y_t).size > 1:
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc_value = float(auc(fpr, tpr))
        ax.plot(fpr, tpr, linewidth=2.0, label=f"ROC (AUC={auc_value:.3f})")
    else:
        ax.text(0.6, 0.2, "AUC=N/A (single class)", transform=ax.transAxes)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(
    history_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Training History",
) -> None:
    """Plot training history curves (loss).

    Args:
        history_df: DataFrame containing at least one loss column. Common columns:
            - train_loss or loss
            - val_loss / validation_loss
            - test_loss
            Optionally contains an 'epoch' column.
        output_path: Where to save the PNG.
        title: Plot title.
    """

    out = _ensure_path(output_path)
    if history_df is None or len(history_df) == 0:
        raise ValueError("history_df is empty")

    df = history_df.copy()
    if "epoch" in df.columns:
        x = pd.to_numeric(df["epoch"], errors="coerce")
    else:
        x = pd.Series(np.arange(1, len(df) + 1), name="epoch")

    # Try a few common conventions
    train_col = "train_loss" if "train_loss" in df.columns else ("loss" if "loss" in df.columns else None)
    val_col = None
    for c in ["val_loss", "valid_loss", "validation_loss"]:
        if c in df.columns:
            val_col = c
            break
    test_col = "test_loss" if "test_loss" in df.columns else None

    if train_col is None and val_col is None and test_col is None:
        raise ValueError(
            "history_df must contain at least one of: train_loss/loss, val_loss/validation_loss, test_loss"
        )

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if train_col is not None:
        ax.plot(x, pd.to_numeric(df[train_col], errors="coerce"), label=train_col)
    if val_col is not None:
        ax.plot(x, pd.to_numeric(df[val_col], errors="coerce"), label=val_col)
    if test_col is not None:
        ax.plot(x, pd.to_numeric(df[test_col], errors="coerce"), label=test_col)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Create a bar chart comparing a single metric across models.

    Args:
        comparison_df: DataFrame with one row per model; expects a 'model' column if present.
        metric: Column name for the metric to plot (e.g. 'roc_auc', 'f1').
        output_path: Where to save the PNG.
        title: Optional plot title.
    """

    out = _ensure_path(output_path)
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric {metric!r} not found in comparison_df columns={list(comparison_df.columns)}")

    df = comparison_df.copy()
    models = df["model"].astype(str) if "model" in df.columns else df.index.astype(str)
    values = pd.to_numeric(df[metric], errors="coerce")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(models, values)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by Model")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(
    scores: Iterable[float] | np.ndarray,
    output_path: str | Path,
    title: str = "Score Distribution",
) -> None:
    """Plot a histogram of anomaly/risk scores.

    Args:
        scores: Iterable of score floats.
        output_path: Where to save the PNG.
        title: Plot title.
    """

    out = _ensure_path(output_path)
    arr = np.asarray(list(scores), dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(arr, bins=30)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Backward-compatible helper (used by older scripts in this repo) ---
def plot_history(history: Any, out_path: str | Path) -> None:
    """Legacy wrapper that accepts a TrainHistory-like object.

    The older training scripts in this repository pass an object with attributes like
    'train_loss' and 'val_loss'. This wrapper converts it into a DataFrame and calls
    plot_training_history.
    """

    data: dict[str, Any] = {}
    for k in ["train_loss", "val_loss", "test_loss", "loss", "validation_loss", "valid_loss"]:
        if hasattr(history, k):
            data[k] = getattr(history, k)
    if not data:
        raise ValueError("history object does not expose recognizable loss attributes")
    plot_training_history(pd.DataFrame(data), out_path)
