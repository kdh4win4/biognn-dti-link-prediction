"""
Train DTI GNN model and save checkpoint.
This file was intentionally minimal and now fixed.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .model import DTI_GNN
from .data import build_graph


def main():
    print("[INFO] Building graph...")
    data: Data = build_graph()

    print(f"[INFO] Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}")

    model = DTI_GNN(in_channels=16, hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("[INFO] Starting training...")
    model.train()

    for epoch in range(1, 51):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, torch.zeros_like(out))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    print("[INFO] Saving model...")
    torch.save(model.state_dict(), "outputs/gnn_model.pt")
    print("[DONE] Model saved to outputs/gnn_model.pt")


if __name__ == "__main__":
    main()
