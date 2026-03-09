from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

@dataclass
class SeqBatch:
    x: torch.Tensor  # [B, T, F]
    y: torch.Tensor  # labels for last event (optional)

class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_last: np.ndarray | None = None):
        self.X_seq = X_seq.astype(np.float32)
        self.y_last = None if y_last is None else y_last.astype(np.int64)

    def __len__(self): return self.X_seq.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X_seq[idx])
        if self.y_last is None:
            y = torch.tensor(-1, dtype=torch.long)
        else:
            y = torch.tensor(self.y_last[idx], dtype=torch.long)
        return x, y

class TransformerAutoencoder(nn.Module):
    def __init__(self, feat_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj_in = nn.Linear(feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        dec_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=max(1, num_layers))
        self.proj_out = nn.Linear(d_model, feat_dim)

    def forward(self, x):
        z = self.proj_in(x)
        z = self.encoder(z)
        z = self.decoder(z)
        x_hat = self.proj_out(z)
        return x_hat

def build_sequences(df: pd.DataFrame, feature_cols: list[str], seq_len: int = 20, group_col: str = "src_ip"):
    """Window sequences per actor (src_ip). Uses the last event label as the window label (if exists)."""
    X_list, y_list, idx_list = [], [], []
    for actor, g in df.groupby(group_col, sort=False):
        g = g.sort_values("timestamp")
        X = g[feature_cols].to_numpy(dtype=float)
        y = g["label"].to_numpy(dtype=int) if "label" in g.columns else None

        if len(g) < seq_len:
            continue
        for i in range(seq_len-1, len(g)):
            window = X[i-seq_len+1:i+1]
            X_list.append(window)
            idx_list.append(g.index[i])  # align score to last event
            if y is not None:
                y_list.append(y[i])
    X_seq = np.stack(X_list) if X_list else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
    y_last = np.array(y_list, dtype=int) if y_list else None
    return X_seq, y_last, np.array(idx_list, dtype=int)

def train_autoencoder(X_seq: np.ndarray, cfg, device: str = "cpu"):
    model = TransformerAutoencoder(
        feat_dim=X_seq.shape[-1],
        d_model=cfg.ae_d_model,
        nhead=cfg.ae_nhead,
        num_layers=cfg.ae_layers,
        dropout=cfg.ae_dropout
    ).to(device)

    ds = SeqDataset(X_seq)
    dl = DataLoader(ds, batch_size=cfg.ae_batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ae_lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(cfg.ae_epochs):
        total = 0.0
        for x, _ in dl:
            x = x.to(device)
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"[AE] epoch {epoch+1}/{cfg.ae_epochs} loss={total/len(ds):.6f}")
    return model

@torch.no_grad()
def score_autoencoder(model: nn.Module, X_seq: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    ds = SeqDataset(X_seq)
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    scores = []
    for x, _ in dl:
        x = x.to(device)
        x_hat = model(x)
        # mean squared error per sequence
        err = torch.mean((x_hat - x) ** 2, dim=(1,2))
        scores.append(err.detach().cpu().numpy())
    return np.concatenate(scores) if scores else np.array([], dtype=float)
