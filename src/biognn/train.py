"""
Train a minimal DTI-GNN model and save a checkpoint.

Why this exists:
- scripts/02_train.sh runs: `python -m src.biognn.train`
- Therefore this module must have an executable entrypoint (main()).

This is a small, portfolio-friendly training loop:
- builds a tiny demo graph (from build_graph)
- trains for a few epochs
- prints loss
- saves weights to outputs/gnn_model.pt
"""

import os
import torch
import torch.nn.functional as F

from .model import DTI_GNN
from .data import build_graph


def main(epochs: int = 50, lr: float = 1e-3) -> None:
    print("[INFO] Building demo graph...")
    data = build_graph()
    print(f"[INFO] Graph: num_nodes={data.num_nodes}, num_edges={data.edge_index.size(1)}")

    # Ensure we have node features.
    # If build_graph() already provides x, keep it.
    # Otherwise, create simple learnable features (random init).
    if getattr(data, "x", None) is None:
        in_channels = 16
        data.x = torch.randn((data.num_nodes, in_channels), dtype=torch.float32)
    else:
        in_channels = data.x.size(-1)

    model = DTI_GNN(in_channels=in_channels, hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("[INFO] Starting training...")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        # Portfolio demo objective:
        # make outputs small (near zero) to show a stable training loop.
        loss = F.mse_loss(out, torch.zeros_like(out))

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    os.makedirs("outputs", exist_ok=True)
    ckpt_path = "outputs/gnn_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[DONE] Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
