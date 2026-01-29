from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MAX

from src.models.gat_encoder import GATEncoder


def _maybe_add_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    return tensor


def _build_batched_graph(
    node_x: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    node_x = _maybe_add_batch(node_x)
    edge_attr = _maybe_add_batch(edge_attr)
    batch_size, num_nodes, _ = node_x.shape
    _, num_edges, _ = edge_attr.shape

    node_x_flat = node_x.reshape(batch_size * num_nodes, -1)
    edge_attr_flat = edge_attr.reshape(batch_size * num_edges, -1)

    device = node_x.device
    edge_index = edge_index.to(device)
    edge_index_batched = edge_index.repeat(1, batch_size)
    offsets = (torch.arange(batch_size, device=device) * num_nodes).repeat_interleave(num_edges)
    edge_index_batched = edge_index_batched + offsets.unsqueeze(0)

    batch_vec = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
    return node_x_flat, edge_attr_flat, edge_index_batched, batch_vec, batch_size, num_edges


class GATMaskedPolicyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cfg = model_config.get("custom_model_config", {})
        node_in = int(cfg["node_in"])
        edge_in = int(cfg["edge_in"])
        hidden = int(cfg.get("hidden", 256))
        embed = int(cfg.get("embed", 256))
        num_layers = int(cfg.get("num_layers", 3))
        self.edge_index = torch.tensor(cfg["edge_index"], dtype=torch.long)

        self.node_norm = nn.LayerNorm(node_in)
        self.edge_norm = nn.LayerNorm(edge_in)
        self.encoder = GATEncoder(node_in, hidden, embed, edge_dim=edge_in, num_layers=num_layers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed * 4 + edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(embed * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        action_mask = obs["action_mask"]
        node_x = obs["obs"]["node_features"]
        edge_attr = obs["obs"]["edge_features"]

        if not torch.is_tensor(action_mask):
            action_mask = torch.as_tensor(action_mask, dtype=torch.float32)
        node_x = torch.as_tensor(node_x, dtype=torch.float32)
        edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32)

        node_x = self.node_norm(node_x)
        edge_attr = self.edge_norm(edge_attr)

        node_x_flat, edge_attr_flat, edge_index_batched, batch_vec, batch_size, num_edges = _build_batched_graph(
            node_x, edge_attr, self.edge_index
        )
        node_emb, global_ctx, _ = self.encoder(node_x_flat, edge_index_batched, edge_attr_flat, batch_vec)
        src, dst = edge_index_batched
        edge_batch = batch_vec[edge_index_batched[0]]
        ctx = global_ctx[edge_batch]
        edge_emb = torch.cat([node_emb[src], node_emb[dst], edge_attr_flat, ctx], dim=1)
        logits = self.edge_mlp(edge_emb).squeeze(-1)
        logits = logits.view(batch_size, num_edges)

        mask = torch.as_tensor(action_mask, device=logits.device, dtype=torch.float32)
        logits = logits.masked_fill(mask <= 0, -FLOAT_MAX)

        self._last_value = self.value_mlp(global_ctx).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._last_value


class GATMaskedQModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cfg = model_config.get("custom_model_config", {})
        node_in = int(cfg["node_in"])
        edge_in = int(cfg["edge_in"])
        hidden = int(cfg.get("hidden", 256))
        embed = int(cfg.get("embed", 256))
        num_layers = int(cfg.get("num_layers", 3))
        self.edge_index = torch.tensor(cfg["edge_index"], dtype=torch.long)

        self.node_norm = nn.LayerNorm(node_in)
        self.edge_norm = nn.LayerNorm(edge_in)
        self.encoder = GATEncoder(node_in, hidden, embed, edge_dim=edge_in, num_layers=num_layers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed * 4 + edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        action_mask = obs["action_mask"]
        node_x = obs["obs"]["node_features"]
        edge_attr = obs["obs"]["edge_features"]

        if not torch.is_tensor(action_mask):
            action_mask = torch.as_tensor(action_mask, dtype=torch.float32)
        node_x = torch.as_tensor(node_x, dtype=torch.float32)
        edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32)

        node_x = self.node_norm(node_x)
        edge_attr = self.edge_norm(edge_attr)

        node_x_flat, edge_attr_flat, edge_index_batched, batch_vec, batch_size, num_edges = _build_batched_graph(
            node_x, edge_attr, self.edge_index
        )
        node_emb, global_ctx, _ = self.encoder(node_x_flat, edge_index_batched, edge_attr_flat, batch_vec)
        src, dst = edge_index_batched
        edge_batch = batch_vec[edge_index_batched[0]]
        ctx = global_ctx[edge_batch]
        edge_emb = torch.cat([node_emb[src], node_emb[dst], edge_attr_flat, ctx], dim=1)
        q_values = self.edge_mlp(edge_emb).squeeze(-1)
        q_values = q_values.view(batch_size, num_edges)

        mask = torch.as_tensor(action_mask, device=q_values.device, dtype=torch.float32)
        q_values = q_values.masked_fill(mask <= 0, -FLOAT_MAX)
        return q_values, state
