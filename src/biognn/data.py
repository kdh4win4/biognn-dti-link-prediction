import torch
from torch_geometric.data import Data

def build_graph(edge_index, num_nodes):
    return Data(edge_index=edge_index, num_nodes=num_nodes)
