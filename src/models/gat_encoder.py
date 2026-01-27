from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class GATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        edge_dim: int,
        heads: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_layers = max(2, num_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim))
        for _ in range(self.num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim))
        self.layers.append(GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, edge_dim=edge_dim))
        self.input_proj = nn.Linear(in_dim, hidden_dim * heads)
        self.norms = nn.ModuleList()
        for i in range(self.num_layers):
            out_dim_i = out_dim if i == self.num_layers - 1 else hidden_dim * heads
            self.norms.append(nn.LayerNorm(out_dim_i))

    def forward(self, x, edge_index, edge_attr, batch, return_attention: bool = False):
        attn = None
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1 and return_attention:
                x, attn_info = layer(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                attn = attn_info[1]
            elif i == len(self.layers) - 1:
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                x_in = x
                x = layer(x, edge_index, edge_attr=edge_attr)
                if i == 0:
                    x_in = self.input_proj(x_in)
                x = self.norms[i](x)
                x = torch.relu(x + x_in)
                continue
            x = self.norms[i](x)
        g_mean = global_mean_pool(x, batch)
        g_max = global_max_pool(x, batch)
        global_ctx = torch.cat([g_mean, g_max], dim=1)
        return x, global_ctx, attn
