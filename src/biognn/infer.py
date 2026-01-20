"""
Inference for the minimal DTI-GNN demo.

Loads:
- the same demo graph used in training
- the saved checkpoint outputs/gnn_model.pt

Prints:
- node embeddings
- a simple link score example (dot product) between two nodes
"""

import torch
from torch_geometric.data import Data

from .model import DTI_GNN
from .data import build_graph


def main(in_channels: int = 16, hidden_dim: int = 32) -> None:
    print("[INFO] Loading demo graph...")

    edge_index = torch.tensor(
        [[0, 1, 2],
         [1, 2, 3]],
        dtype=torch.long
    )
    num_nodes = 4
    data: Data = build_graph(edge_index=edge_index, num_nodes=num_nodes)

    if getattr(data, "x", None) is None:
        data.x = torch.randn((data.num_nodes, in_channels), dtype=torch.float32)

    print(f"[INFO] Graph: num_nodes={data.num_nodes}, num_edges={data.edge_index.size(1)}")

    model = DTI_GNN(in_dim=in_channels, hidden_dim=hidden_dim)

    ckpt_path = "outputs/gnn_model.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        z = model(data.x, data.edge_index)  # node embeddings, shape: [num_nodes, hidden_dim]

    print(f"[INFO] Embeddings shape: {tuple(z.shape)}")

    # Demo "link prediction" score: dot product between node 0 and node 3
    # (In a real DTI setting, you'd compute scores between drug nodes and target nodes.)
    i, j = 0, 3
    score = float((z[i] * z[j]).sum().item())

    print("[RESULT] Example link score")
    print(f"  node_i={i}  node_j={j}  score={score:.4f}")


if __name__ == "__main__":
    main()
