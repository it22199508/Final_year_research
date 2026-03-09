from __future__ import annotations
import numpy as np
from imblearn.over_sampling import SMOTE

def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Balances classes for supervised XGBoost training."""
    sm = SMOTE(random_state=random_state, k_neighbors=min(5, max(1, int(np.sum(y==1)-1))))
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res
