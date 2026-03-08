from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F), lengths: (B,)
        lengths = torch.clamp(lengths, min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        # h_n: (num_layers, B, hidden)
        last = h_n[-1]
        logits = self.head(last).squeeze(-1)
        return logits


class GRUAnomalyDetector(nn.Module):
    """A simple GRU-based anomaly detector that consumes fixed-length sequences.

    This model is intentionally "production-style": it expects a dense tensor
    input and does not require sequence lengths / packing.

    Input:  x of shape (B, T, F)
    Output: logits of shape (B, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, T, F); got {tuple(x.shape)}")
        _, h_n = self.gru(x)
        last = h_n[-1]
        logits = self.head(last)
        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_proba(
    model: nn.Module,
    x_tensor: torch.Tensor,
    device: str | torch.device = "cpu",
) -> "Any":
    """Return predicted probabilities as a 1D NumPy array of shape (B,).

    This helper expects `model(x_tensor)` to return logits.
    """

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        logits = model(x_tensor.to(device))
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs.detach().cpu().numpy()


ModelT = TypeVar("ModelT", bound=nn.Module)


def save_model_artifacts(
    model: nn.Module,
    output_dir: str | Path,
    *,
    model_filename: str = "model.pt",
    meta_filename: str = "meta.json",
    metadata: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save model weights + a small JSON metadata file.

    Returns `(model_path, meta_path)`.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / model_filename
    meta_path = out_dir / meta_filename

    torch.save(model.state_dict(), model_path)

    meta: dict[str, Any] = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "pytorch_version": torch.__version__,
        "model_class": model.__class__.__name__,
    }
    if model_config is not None:
        meta["model_config"] = model_config
    if metadata:
        meta.update(metadata)

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return model_path, meta_path


def load_model_artifacts(
    model_class: type[ModelT],
    artifacts_path: str | Path,
    *,
    model_filename: str = "model.pt",
    meta_filename: str = "meta.json",
    model_kwargs: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> ModelT:
    """Load model weights from a directory (or explicit weight file path)."""

    device = torch.device(device)
    p = Path(artifacts_path)
    if p.is_dir():
        weights_path = p / model_filename
        meta_path = p / meta_filename
    else:
        weights_path = p
        meta_path = p.with_name(meta_filename)

    kwargs = dict(model_kwargs or {})
    model = model_class(**kwargs).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Optional: validate class name if metadata exists.
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            saved_class = meta.get("model_class")
            if saved_class and saved_class != model_class.__name__:
                raise ValueError(
                    f"Metadata model_class={saved_class!r} does not match {model_class.__name__!r}"
                )
        except json.JSONDecodeError:
            pass

    return model


if __name__ == "__main__":
    import tempfile

    torch.manual_seed(0)

    B, T, F = 4, 12, 8
    x = torch.randn(B, T, F)

    m = GRUAnomalyDetector(input_size=F, hidden_size=32)
    print("params:", count_trainable_parameters(m))
    p1 = predict_proba(m, x)
    print("probs shape:", p1.shape)

    tmp_dir = Path(tempfile.mkdtemp(prefix="insider_gru_demo_"))
    save_model_artifacts(
        m,
        tmp_dir,
        metadata={"note": "demo save/load"},
        model_config={"input_size": F, "hidden_size": 32, "num_layers": 1, "dropout": 0.2},
    )
    m2 = load_model_artifacts(GRUAnomalyDetector, tmp_dir, model_kwargs={"input_size": F, "hidden_size": 32})
    p2 = predict_proba(m2, x)
    print("max |p1-p2|:", float(torch.tensor(p1 - p2).abs().max()))
