from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Centralized dataset configuration.

    This project uses CERT-style CSV event logs (email, device/login, etc.) stored under
    the `Dataset/` directory. The training pipeline consumes one or more CSV files matched
    by glob patterns.

        Notes on the dataset layout used by this repository:
        - The workspace is expected to contain fixed split folders:
                Dataset/train/R1 .. Dataset/train/R4
                Dataset/test/R1 .. Dataset/test/R4
        - Training scripts should load only from Dataset/train/*.
        - Evaluation scripts should load only from Dataset/test/*.
        - Filenames may still contain legacy tokens (e.g., device-*-train-data.csv) but the
            *folder split* is authoritative.
    """

    # --- Paths / directories (requested) ---
    raw_email_data_path: Path = Path("Dataset")
    raw_login_data_path: Path = Path("Dataset")
    processed_data_dir: Path = Path("outputs") / "processed"
    labels_dir: Path = Path(".")

    # --- Dataset root + globs (backward compatible with existing scripts) ---
    dataset_root: Path = Path("Dataset")

    # Backward-compatible globs (kept for scripts that still use globbing).
    # Prefer using `insider_gru.data.load_combined_event_logs_from_split(...)`.
    train_glob: str = "train/R*/email-*-data*.csv"
    test_glob: str = "test/R*/email-*-data*.csv"

    # --- Schema ---
    timestamp_col: str = "date"
    id_col: str = "id"

    # Optional dataset-specific columns (useful for future pipelines)
    email_sender_col: str = "from"
    email_recipient_col: str = "to"
    email_content_col: str = "content"
    email_attachment_count_col: str = "attachments"
    email_size_col: str = "size"

    login_user_col: str = "user"
    login_device_col: str = "pc"
    login_activity_col: str = "activity"

    # --- Labels ---
    # If your CSV includes a label column, set this.
    label_col: str | None = None

    # Alternative: an id->label CSV with columns: id,label
    # Common defaults in this repo: labels_from_rules.csv, labels_from_users.csv, labels_template.csv
    label_mapping_csv: Path | None = Path("labels_from_rules.csv")

    # Sequence building (grouping)
    group_col_candidates: tuple[str, ...] = ("user", "from")

    # Feature selection:
    # - "minimal": engineered numeric features only
    # - "all": include additional columns (may expand dimensionality)
    feature_mode: str = "minimal"

    def resolved_label_mapping_path(self) -> Path | None:
        """Return an absolute-ish path for label mapping if configured."""
        if self.label_mapping_csv is None:
            return None
        p = Path(self.label_mapping_csv)
        if p.is_absolute():
            return p
        # Prefer labels_dir when set.
        return (Path(self.labels_dir) / p).resolve()


@dataclass
class GRUConfig:
    """Model configuration for the GRU baseline.

    The training pipeline currently infers `input_dim` from the fitted preprocessor.
    Keep `input_dim=None` unless you implement a fixed feature schema.
    """

    input_dim: int | None = None
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.2

    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 30
    sequence_length: int = 20


@dataclass
class TransformerConfig:
    """Model configuration for the (planned) Transformer sequence model."""

    input_dim: int | None = None
    # Defaults are intentionally modest for small behavioral sequences.
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1

    learning_rate: float = 3e-4
    batch_size: int = 128
    num_epochs: int = 20
    sequence_length: int = 20


@dataclass
class ScoringConfig:
    """Risk scoring configuration used by dashboards and reporting.

    This config supports:
    - Model Prediction: use the model probability as the score.
    - Manual Scoring: compute score from configurable rules.
    - Hybrid: blend model probability and manual score.
    """

    # Decision threshold for alerting
    threshold: float = 0.50

    # "Model Prediction" | "Manual Scoring" | "Hybrid"
    scoring_mode: str = "Hybrid"

    # Hybrid blend weight: fraction from model probability (0..1)
    hybrid_alpha: float = 0.50

    # Optional base risk to add before rule contributions (0..1)
    base_risk_score: float = 0.10

    # --- Manual weights (0..1) ---
    # Email-focused rules
    after_hours_weight: float = 0.20
    large_content_weight: float = 0.20
    attachment_weight: float = 0.15
    external_recipient_weight: float = 0.15
    suspicious_keyword_weight: float = 0.20

    # Login-focused rules
    weekend_login_weight: float = 0.20
    high_login_frequency_weight: float = 0.20
    unusual_device_weight: float = 0.20

    # --- Thresholds ---
    content_size_threshold: int = 1_000_000
    attachment_threshold: int = 3
    login_frequency_threshold: int = 5
    business_hours_start: int = 8
    business_hours_end: int = 18

    suspicious_keywords: list[str] = field(default_factory=lambda: ["password", "confidential", "privileged", "admin"])


@dataclass
class IntegrityConfig:
    """Alert integrity / ledger configuration (planned extension).

    Intended use: store cryptographic hashes of exported alerts/reports to detect tampering.
    """

    enabled: bool = True
    ledger_path: Path = Path("outputs") / "integrity" / "ledger.jsonl"
    hash_algorithm: str = "sha256"


@dataclass
class StreamingConfig:
    """Real-time/near-real-time simulation configuration (planned extension)."""

    enabled: bool = False
    source_file: Path = Path("Dataset")
    delay_seconds: float = 0.25

    # Default locations used by scripts/simulate_stream.py and apps/streamlit_app.py
    live_events_path: Path = Path("outputs") / "live" / "live_events.jsonl"
    live_alerts_path: Path = Path("outputs") / "live" / "live_alerts.jsonl"


@dataclass
class TrainConfig:
    """Backward-compatible training config for the existing GRU training pipeline.

    This class is already used by scripts and dashboards in this repository.
    Keep field names stable.
    """

    seq_len: int = 20
    stride: int = 1
    label_strategy: str = "last"  # any_positive | last
    window_anchor: str = "end"  # start | end

    batch_size: int = 256
    max_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    hidden_size: int = 64
    num_layers: int = 1

    early_stopping_patience: int = 5
    val_split: float = 0.2
    seed: int = 42

    threshold: float = 0.5


@dataclass
class OutputConfig:
    """Output directories for artifacts, reports, and plots."""

    out_dir: Path = Path("outputs")
    model_dir: Path = Path("outputs/model")
    plots_dir: Path = Path("outputs/plots")
    reports_dir: Path = Path("outputs/reports")

    # Additional outputs (requested)
    integrity_dir: Path = Path("outputs") / "integrity"


@dataclass
class AppConfig:
    """Top-level application configuration.

    This groups all sub-configs so you can pass around a single object.
    """

    data: DataConfig = field(default_factory=DataConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Keep the existing training configuration accessible.
    train: TrainConfig = field(default_factory=TrainConfig)


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert dataclass config to a JSON-serializable dict.

    - Dataclasses are converted recursively.
    - `pathlib.Path` values become strings.
    """

    def _convert(x: Any) -> Any:
        if is_dataclass(x):
            return {k: _convert(v) for k, v in asdict(x).items()}
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {str(k): _convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_convert(v) for v in x]
        return x

    converted = _convert(cfg)
    if isinstance(converted, dict):
        return converted
    return {"value": converted}


def get_default_config() -> AppConfig:
    """Return a fully-populated default configuration.

    This is a convenience helper for apps/scripts that want a single entry point.
    """

    app = AppConfig()

    # Keep defaults aligned across sections (minimal, non-invasive mapping).
    # GRUConfig mirrors TrainConfig defaults.
    app.gru = GRUConfig(
        input_dim=None,
        hidden_dim=int(app.train.hidden_size),
        num_layers=int(app.train.num_layers),
        dropout=float(app.train.dropout),
        learning_rate=float(app.train.lr),
        batch_size=int(app.train.batch_size),
        num_epochs=int(app.train.max_epochs),
        sequence_length=int(app.train.seq_len),
    )

    # OutputConfig defaults should remain stable.

    # TransformerConfig defaults: keep comparable with GRU by aligning the
    # default sequence length to the GRU/TrainConfig seq_len.
    app.transformer = TransformerConfig(
        input_dim=None,
        d_model=int(app.transformer.d_model),
        nhead=int(app.transformer.nhead),
        num_layers=int(app.transformer.num_layers),
        dim_feedforward=int(app.transformer.dim_feedforward),
        dropout=float(app.transformer.dropout),
        learning_rate=float(app.transformer.learning_rate),
        batch_size=int(app.transformer.batch_size),
        num_epochs=int(app.transformer.num_epochs),
        sequence_length=int(app.train.seq_len),
    )
    return app
