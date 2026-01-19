import torch
from .model import DTI_GNN

def infer(model, data):
    model.eval()
    with torch.no_grad():
        return model(data.x, data.edge_index)
