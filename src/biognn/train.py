"""
Train a minimal Drug–Target Interaction (DTI) GNN model and save checkpoint.

This is a portfolio-grade, fully executable training script:
- builds a small demo graph
- runs a stable training loop
- prints progress
- saves model weights for inference

Executed via:
    python -m src.biognn.train
"""

import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .model import DTI_GNN
from .data import build_graph


def main(
    epochs: int = 50,
    lr: float = 1e-3,
    in_channels: int = 16,
    hidden_channels: int = 32,
):
    print("[INFO] Building demo graph...")

    # ------------------------------------------------------------------
    # Minimal demo graph (same scale as scripts/01_build_graph.py)
    # ------------------------------------------------------------------
    edge_index = torch.tensor(
        [[0, 1, 2],
         [1, 2, 3]],
        dtype=torch.long
    )
    num_nodes = 4

    data: Data = build_graph(edge_index=edge_index, num_nodes=num_nodes)

    print(f"[INFO] Graph: num_nodes={data.num_nodes}, num_edges={data.edge_index.size(1)}")

    # ------------------------------------------------------------------
    # Node features (random init for demo / portfolio purposes)
    # ------------------------------------------------------------------
    if getattr(data, "x", None) is None:
        data.x = torch.randn((data.num_nodes, in_channels), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = DTI_GNN(in_dim=in_channels, hidden_dim=hidden_channels)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("[INFO] Starting training...")
    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        # Simple stable objective for demo purposes
        loss = F.mse_loss(out, torch.zeros_like(out))

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    ckpt_path = "outputs/gnn_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    print(f"[DONE] Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
