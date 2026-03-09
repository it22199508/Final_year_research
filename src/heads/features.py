from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer

def _extract_url_host(url: str) -> str:
    if not url:
        return ""
    m = re.match(r"^https?://([^/]+)/?", url)
    return m.group(1).lower() if m else ""

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour.fillna(0).astype(int)
    out["dow"] = out["timestamp"].dt.dayofweek.fillna(0).astype(int)
    out["url_host"] = out["url"].apply(_extract_url_host)
    out["bytes_total"] = (out["bytes_sent"] + out["bytes_received"]).astype(int)
    out["is_web"] = out["dst_port"].isin([80, 443]).astype(int)
    return out

class TabularFeaturizer:
    """Produces a numeric feature matrix for XGBoost.

    - Categorical: protocol, url_host, is_internal_traffic, dst_port bucket
    - Text (optional): hashed bag of words from URL path + user_agent
    - Numeric: ports, byte counts, hour, day-of-week
    """
    def __init__(self, use_text: bool = True, text_dim: int = 64):
        self.use_text = use_text
        self.text_dim = text_dim
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.hv = HashingVectorizer(n_features=text_dim, alternate_sign=False, norm=None)
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        cat = df[["protocol","url_host","is_internal_traffic"]].astype(str)
        self.ohe.fit(cat)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        num = df[["src_port","dst_port","bytes_sent","bytes_received","bytes_total","hour","dow","is_web"]].to_numpy(dtype=float)

        cat = df[["protocol","url_host","is_internal_traffic"]].astype(str)
        cat_mat = self.ohe.transform(cat)

        if self.use_text:
            text = (df["url"].fillna("") + " " + df["user_agent"].fillna("")).astype(str).tolist()
            text_mat = self.hv.transform(text).toarray().astype(float)
            X = np.hstack([num, cat_mat, text_mat])
        else:
            X = np.hstack([num, cat_mat])
        return X
