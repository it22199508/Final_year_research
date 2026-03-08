from __future__ import annotations

import math
from typing import Union

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding.

    Compatible with `batch_first=True` tensors of shape (B, T, D).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, D) input; got {tuple(x.shape)}")
        t = x.size(1)
        x = x + self.pe[:t, :].unsqueeze(0)
        return self.dropout(x)


class TransformerAnomalyDetector(nn.Module):
    """A small Transformer encoder for fixed-length behavioral sequences.

    Input:  x of shape (B, T, F)
    Output: logits of shape (B, 1)

    Design goals:
    - Minimal and comparable to GRU baseline
    - Fixed-length inputs (no lengths / packing)
    - Reusable in scripts with the same training utilities
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(int(input_dim), int(d_model))
        self.pos_enc = PositionalEncoding(d_model=int(d_model), dropout=float(dropout))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Logits tensor of shape (batch_size, 1)
        """

        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, T, F); got {tuple(x.shape)}")

        z = self.input_proj(x)  # (B, T, D)
        z = self.pos_enc(z)
        z = self.encoder(z)  # (B, T, D)

        pooled = z.mean(dim=1)
        return self.head(pooled)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_proba(
    model: nn.Module,
    x_tensor: torch.Tensor,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Predict probabilities for a batch.

    Applies `sigmoid` to model logits.

    Args:
        model: A PyTorch model returning logits.
        x_tensor: Input tensor of shape (B, T, F).
        device: Torch device.

    Returns:
        Probabilities as a CPU tensor of shape (B,).
    """

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    with torch.inference_mode():
        logits = model(x_tensor.to(dev))
        probs = torch.sigmoid(logits).view(-1)
        return probs.detach().cpu()


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, F = 4, 20, 12
    x = torch.randn(B, T, F)

    m = TransformerAnomalyDetector(input_dim=F, d_model=64, nhead=4, num_layers=2)
    print("params:", count_trainable_parameters(m))

    probs = predict_proba(m, x)
    print("probs shape:", tuple(probs.shape))
    print("probs sample:", probs[:3].tolist())
