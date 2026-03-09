from __future__ import annotations
import numpy as np
from xgboost import XGBClassifier

def train_xgboost(X: np.ndarray, y: np.ndarray, cfg):
    model = XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_lr,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(X, y)
    return model
