from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

try:
    import torch
    from torch_geometric.data import Data
except Exception:
    Data = None
    torch = None

@dataclass
class GraphArtifacts:
    data: "Data"
    node_id_map: Dict[Tuple[str,str], int]  # (type, key) -> node_id
    node_type: np.ndarray  # node_type id per node

def _get_node(node_map, node_types, t: str, key: str):
    k = (t, key)
    if k in node_map:
        return node_map[k]
    nid = len(node_map)
    node_map[k] = nid
    node_types.append(t)
    return nid

def build_graph(df: pd.DataFrame):
    """Builds a single homogeneous graph with typed nodes:
    - actor: src_ip
    - resource: dst_ip
    - host: url_host
    - proto: protocol
    Edges connect actor→resource, actor→host, actor→proto.
    """
    if Data is None:
        raise ImportError("torch_geometric is required for graph components. Install torch-geometric.")

    node_map = {}
    node_types = []
    edges = []

    # ensure url_host exists
    if "url_host" not in df.columns:
        df = df.copy()
        df["url_host"] = ""

    for _, r in df.iterrows():
        a = _get_node(node_map, node_types, "actor", str(r["src_ip"]))
        b = _get_node(node_map, node_types, "resource", str(r["dst_ip"]))
        edges.append((a, b))

        h = str(r.get("url_host",""))
        if h and h != "nan":
            hh = _get_node(node_map, node_types, "host", h)
            edges.append((a, hh))

        p = _get_node(node_map, node_types, "proto", str(r["protocol"]))
        edges.append((a, p))

    edge_index = np.array(edges, dtype=np.int64).T  # [2, E]

    # Node features: simple type embedding + degree features
    import torch
    num_nodes = len(node_map)
    type_list = sorted(list(set(node_types)))
    type_to_id = {t:i for i,t in enumerate(type_list)}
    node_type_id = np.array([type_to_id[t] for t in node_types], dtype=np.int64)

    # degree
    deg = np.zeros(num_nodes, dtype=np.float32)
    for u, v in edges:
        deg[u] += 1.0
        deg[v] += 1.0
    x = np.stack([node_type_id.astype(np.float32), np.log1p(deg)], axis=1)  # [N,2]

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        num_nodes=num_nodes
    )
    return GraphArtifacts(data=data, node_id_map=node_map, node_type=node_type_id)
