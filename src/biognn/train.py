import torch
from .model import DTI_GNN

def train(model, data, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = out.mean()
        loss.backward()
        optimizer.step()
