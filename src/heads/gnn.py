from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
except Exception:
    SAGEConv = None

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for GraphSAGE. Install torch-geometric.")
        self.convs = nn.ModuleList()
        if layers == 1:
            self.convs.append(SAGEConv(in_dim, hidden))
        else:
            self.convs.append(SAGEConv(in_dim, hidden))
            for _ in range(layers-1):
                self.convs.append(SAGEConv(hidden, hidden))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x

def _negative_sampling(edge_index: torch.Tensor, num_nodes: int, num_neg: int):
    # Very simple negative sampling (uniform)
    src = torch.randint(0, num_nodes, (num_neg,), device=edge_index.device)
    dst = torch.randint(0, num_nodes, (num_neg,), device=edge_index.device)
    return torch.stack([src, dst], dim=0)

def train_graphsage_linkpred(data, cfg, device: str = "cpu"):
    """Unsupervised relational modelling via link prediction reconstruction.
    Produces node embeddings; relational anomaly score is based on edge likelihood.
    """
    model = GraphSAGEEncoder(in_dim=data.x.size(1), hidden=cfg.gnn_hidden, layers=cfg.gnn_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.gnn_lr)
    data = data.to(device)

    E = data.edge_index.size(1)
    num_nodes = data.num_nodes
    pos_edge_index = data.edge_index

    model.train()
    for epoch in range(cfg.gnn_epochs):
        z = model(data.x, data.edge_index)
        # positive scores (dot product)
        src, dst = pos_edge_index
        pos_score = (z[src] * z[dst]).sum(dim=1)

        # negative edges
        neg_edge_index = _negative_sampling(pos_edge_index, num_nodes, num_neg=E)
        ns, nd = neg_edge_index
        neg_score = (z[ns] * z[nd]).sum(dim=1)

        loss = -torch.mean(F.logsigmoid(pos_score)) - torch.mean(F.logsigmoid(-neg_score))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[GNN] epoch {epoch+1}/{cfg.gnn_epochs} loss={loss.item():.6f}")
    return model

@torch.no_grad()
def edge_anomaly_scores(model, data, edge_index: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """Lower edge likelihood => higher anomaly score."""
    model.eval()
    data = data.to(device)
    z = model(data.x, data.edge_index)
    edge_index = edge_index.to(device)
    src, dst = edge_index
    score = (z[src] * z[dst]).sum(dim=1)
    # convert to anomaly: -logsigmoid(score)
    anom = (-torch.log(torch.sigmoid(score) + 1e-9)).detach().cpu().numpy()
    return anom
