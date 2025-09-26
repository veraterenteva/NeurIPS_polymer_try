import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GINEConv, BatchNorm, GlobalAttention

class EdgeMLP(nn.Module):
    def __init__(self, edge_in, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(edge_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
    def forward(self, e): return self.mlp(e)

class GNNRegressor(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_layers, n_tasks, dropout=0.15):
        super().__init__()
        self.node_embed = nn.Linear(node_in, hidden)
        self.edge_embed = EdgeMLP(edge_in, hidden)
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            nn_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(nn_mlp, edge_dim=hidden))
            self.bns.append(BatchNorm(hidden))
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1)
        ))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks)
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr) if edge_attr is not None else None
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, e)
            h = bn(h); h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = self.pool(h, batch)
        return self.head(g)

    def graph_repr(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr) if edge_attr is not None else None
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, e)
            h = bn(h); h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = self.pool(h, batch)
        return g
