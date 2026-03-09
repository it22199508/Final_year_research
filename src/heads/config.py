from dataclasses import dataclass
from pathlib import Path

@dataclass
class HEADSConfig:
    raw_csv: Path
    processed_dir: Path = Path("data/processed")
    model_dir: Path = Path("models")

    # Temporal modelling
    seq_len: int = 20
    ae_d_model: int = 64
    ae_nhead: int = 4
    ae_layers: int = 2
    ae_dropout: float = 0.1
    ae_epochs: int = 8
    ae_batch_size: int = 256
    ae_lr: float = 3e-4

    # Graph modelling
    gnn_hidden: int = 64
    gnn_layers: int = 2
    gnn_epochs: int = 8
    gnn_lr: float = 1e-3
    gnn_batch_size: int = 2048

    # XGBoost
    xgb_max_depth: int = 5
    xgb_n_estimators: int = 400
    xgb_lr: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
