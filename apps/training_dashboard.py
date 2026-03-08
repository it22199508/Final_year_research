from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Model Training and Evaluation", layout="wide")


def _project_root() -> Path:
    # apps/ is expected to live directly under the repository root
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _load_model_comparison(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Be defensive about index-vs-column formats.
    if "model" not in df.columns:
        df = df.reset_index().rename(columns={"index": "model"})

    # Standardize expected metric column names if present with minor variations.
    rename_map: dict[str, str] = {}
    for col in df.columns:
        low = str(col).strip().lower()
        if low in {"roc", "roc-auc", "roc_auc", "roc auc"}:
            rename_map[col] = "roc_auc"
    if rename_map:
        df = df.rename(columns=rename_map)

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if metric in df.columns:
            df[metric] = _coerce_numeric(df[metric])

    df["model"] = df["model"].astype(str)
    return df


def _best_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    if metric not in df.columns:
        return None
    valid = df[["model", metric]].dropna(subset=[metric])
    if valid.empty:
        return None
    idx = valid[metric].idxmax()
    return df.loc[idx]


def _render_metrics_table(df: pd.DataFrame) -> None:
    st.subheader("1. Summary metrics table")
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_best_model_highlight(df: pd.DataFrame) -> None:
    st.subheader("2. Best model highlight")
    col1, col2 = st.columns(2)

    best_f1 = _best_row(df, "f1")
    with col1:
        if best_f1 is None:
            st.info("No usable F1 scores found.")
        else:
            st.metric("Best by F1", str(best_f1.get("model", "-")), f"{best_f1.get('f1'):.3f}")

    best_auc = _best_row(df, "roc_auc")
    with col2:
        if best_auc is None:
            st.info("No usable ROC-AUC scores found.")
        else:
            st.metric("Best by ROC-AUC", str(best_auc.get("model", "-")), f"{best_auc.get('roc_auc'):.3f}")


def _render_plots(final_eval_dir: Path) -> None:
    st.subheader("3. Evaluation plots")

    plot_paths: list[Path] = []
    plots_dir = final_eval_dir / "plots"
    if plots_dir.exists():
        plot_paths.extend(sorted(plots_dir.glob("*.png")))
    plot_paths.extend(sorted(p for p in final_eval_dir.glob("*.png") if p.is_file()))
    plot_paths = [p for i, p in enumerate(plot_paths) if p not in plot_paths[:i]]

    if not plot_paths:
        st.info("No PNG plots found under outputs/final_evaluation/.")
        return

    for p in plot_paths:
        st.image(str(p), caption=str(p.relative_to(_project_root()).as_posix()), use_container_width=True)


def _render_per_model_details(df: pd.DataFrame, project_root: Path) -> None:
    st.subheader("4. Per-model details")

    # Rows are expected to have names like: Logistic Regression, Random Forest, Isolation Forest, GRU, Transformer
    baseline_names = {"logistic regression", "random forest", "isolation forest", "logreg", "rf", "isoforest"}
    gru_names = {"gru"}
    transformer_names = {"transformer"}

    def pick_rows(name_set: set[str]) -> pd.DataFrame:
        mask = df["model"].str.strip().str.lower().isin(name_set)
        return df.loc[mask].copy()

    with st.expander("Baseline models", expanded=False):
        sub = pick_rows(baseline_names)
        if sub.empty:
            st.info("No baseline rows found in the final comparison table.")
        else:
            st.dataframe(sub, use_container_width=True, hide_index=True)

        baseline_candidates = [
            project_root / "outputs" / "baselines" / "login" / "metrics.json",
            project_root / "outputs" / "baselines" / "email" / "metrics.json",
        ]
        found_any = False
        for metrics_path in baseline_candidates:
            if metrics_path.exists():
                found_any = True
                st.caption(str(metrics_path.relative_to(project_root).as_posix()))
                st.json(_read_json(metrics_path))
        if not found_any:
            st.info("Baseline metrics.json not found under outputs/baselines/ (login or email).")

    with st.expander("GRU", expanded=False):
        sub = pick_rows(gru_names)
        if not sub.empty:
            st.dataframe(sub, use_container_width=True, hide_index=True)
        else:
            st.info("No GRU row found in the final comparison table.")

        metrics_path = project_root / "outputs" / "reports" / "metrics.json"
        report_path = project_root / "outputs" / "reports" / "classification_report.txt"
        if metrics_path.exists():
            st.caption(str(metrics_path.relative_to(project_root).as_posix()))
            st.json(_read_json(metrics_path))
        else:
            st.info("GRU metrics.json not found under outputs/reports/.")
        if report_path.exists():
            with st.expander("classification_report.txt", expanded=False):
                st.code(_read_text(report_path), language="text")

    with st.expander("Transformer", expanded=False):
        sub = pick_rows(transformer_names)
        if not sub.empty:
            st.dataframe(sub, use_container_width=True, hide_index=True)
        else:
            st.info("No Transformer row found in the final comparison table.")

        metrics_path = project_root / "outputs" / "transformer" / "reports" / "metrics.json"
        report_path = project_root / "outputs" / "transformer" / "reports" / "classification_report.txt"
        if metrics_path.exists():
            st.caption(str(metrics_path.relative_to(project_root).as_posix()))
            st.json(_read_json(metrics_path))
        else:
            st.info("Transformer metrics.json not found under outputs/transformer/reports/.")
        if report_path.exists():
            with st.expander("classification_report.txt", expanded=False):
                st.code(_read_text(report_path), language="text")


def main() -> None:
    st.title("Model Training and Evaluation Dashboard")

    project_root = _project_root()
    final_eval_dir = project_root / "outputs" / "final_evaluation"
    comparison_csv = final_eval_dir / "model_comparison.csv"

    if not comparison_csv.exists():
        st.warning(
            "Missing outputs/final_evaluation/model_comparison.csv. "
            "Run scripts/run_evaluation.py to generate final evaluation outputs."
        )
        st.caption(f"Expected path: {comparison_csv}")
        st.stop()

    try:
        df = _load_model_comparison(comparison_csv)
    except Exception as e:
        st.error("Failed to load model_comparison.csv")
        st.exception(e)
        st.stop()

    _render_metrics_table(df)
    st.markdown("---")

    _render_best_model_highlight(df)
    st.markdown("---")

    _render_plots(final_eval_dir)
    st.markdown("---")

    _render_per_model_details(df, project_root)


if __name__ == "__main__":
    main()
