from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.num_layers = max(2, num_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True))
        for _ in range(self.num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
        self.layers.append(GATConv(hidden_dim * heads, out_dim, heads=1, concat=False))
        self.input_proj = nn.Linear(in_dim, hidden_dim * heads)

    def forward(self, x, edge_index, batch, return_attention: bool = False):
        attn = None
        x_skip = self.input_proj(x)
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1 and return_attention:
                x, attn_info = layer(x, edge_index, return_attention_weights=True)
                attn = attn_info[1]
            else:
                x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = torch.relu(x + x_skip)
        global_ctx = global_mean_pool(x, batch)
        return x, global_ctx, attn
