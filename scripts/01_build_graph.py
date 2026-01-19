import torch
from src.biognn.data import build_graph

edge_index = torch.tensor([[0,1,2],[1,2,3]])
graph = build_graph(edge_index, num_nodes=4)
print(graph)
