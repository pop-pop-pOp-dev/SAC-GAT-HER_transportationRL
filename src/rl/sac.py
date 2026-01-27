from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.utils import softmax

from src.models.gat_encoder import GATEncoder


@dataclass
class SACOutput:
    action: int
    log_prob: torch.Tensor
    probs: torch.Tensor


class Actor(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int, embed: int, num_layers: int = 3):
        super().__init__()
        self.encoder = GATEncoder(node_in, hidden, embed, edge_dim=edge_in, num_layers=num_layers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed * 4 + edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_x, edge_index, edge_attr, action_mask, batch, return_attention: bool = False):
        node_emb, global_ctx, attn = self.encoder(node_x, edge_index, edge_attr, batch, return_attention=return_attention)
        src, dst = edge_index
        edge_batch = batch[edge_index[0]]
        ctx = global_ctx[edge_batch]
        edge_emb = torch.cat([node_emb[src], node_emb[dst], edge_attr, ctx], dim=1)
        logits = self.edge_mlp(edge_emb).squeeze(-1)
        logits = logits.masked_fill(action_mask <= 0, -1e9)
        probs = softmax(logits, edge_batch)
        return logits, probs, attn


class Critic(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden: int,
        embed: int,
        num_layers: int = 3,
        encoder: GATEncoder | None = None,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else GATEncoder(node_in, hidden, embed, edge_dim=edge_in, num_layers=num_layers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed * 4 + edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_x, edge_index, edge_attr, batch):
        node_emb, global_ctx, _ = self.encoder(node_x, edge_index, edge_attr, batch)
        src, dst = edge_index
        edge_batch = batch[edge_index[0]]
        ctx = global_ctx[edge_batch]
        edge_emb = torch.cat([node_emb[src], node_emb[dst], edge_attr, ctx], dim=1)
        q = self.edge_mlp(edge_emb).squeeze(-1)
        return q


class DiscreteSAC:
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden: int,
        embed: int,
        num_layers: int = 3,
        lr: float = 3e-4,
        actor_lr: float | None = None,
        critic_lr: float | None = None,
        alpha_lr: float | None = None,
        grad_clip: float | None = None,
        gamma: float = 0.99,
        target_tau: float = 0.005,
        target_entropy: float = None,
        alpha_init: float = 0.1,
        share_critic_encoder: bool = True,
    ):
        self.actor = Actor(node_in, edge_in, hidden, embed, num_layers=num_layers)
        self.share_critic_encoder = share_critic_encoder
        if share_critic_encoder:
            self.critic_encoder = GATEncoder(node_in, hidden, embed, edge_dim=edge_in, num_layers=num_layers)
            self.target_encoder = GATEncoder(node_in, hidden, embed, edge_dim=edge_in, num_layers=num_layers)
            self.critic1 = Critic(node_in, edge_in, hidden, embed, num_layers=num_layers, encoder=self.critic_encoder)
            self.critic2 = Critic(node_in, edge_in, hidden, embed, num_layers=num_layers, encoder=self.critic_encoder)
            self.target1 = Critic(node_in, edge_in, hidden, embed, num_layers=num_layers, encoder=self.target_encoder)
            self.target2 = Critic(node_in, edge_in, hidden, embed, num_layers=num_layers, encoder=self.target_encoder)
            self.target_encoder.load_state_dict(self.critic_encoder.state_dict())
        else:
            self.critic1 = Critic(node_in, edge_in, hidden, embed, num_layers=num_layers)
            self.critic2 = Critic(node_in, edge_in, hidden, embed, num_layers=num_layers)
            self.target1 = Critic(node_in, edge_in, hidden, embed, num_layers=num_layers)
            self.target2 = Critic(node_in, edge_in, hidden, embed, num_layers=num_layers)
            self.target1.load_state_dict(self.critic1.state_dict())
            self.target2.load_state_dict(self.critic2.state_dict())

        actor_lr = lr if actor_lr is None else actor_lr
        critic_lr = lr if critic_lr is None else critic_lr
        alpha_lr = lr if alpha_lr is None else alpha_lr
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        if share_critic_encoder:
            critic_params = (
                list(self.critic_encoder.parameters())
                + list(self.critic1.edge_mlp.parameters())
                + list(self.critic2.edge_mlp.parameters())
            )
        else:
            critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, lr=critic_lr)

        self.log_alpha = torch.tensor(float(np.log(max(alpha_init, 1e-8))), requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.gamma = gamma
        self.target_tau = target_tau
        self.target_entropy = target_entropy
        self.grad_clip = grad_clip

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, node_x, edge_index, edge_attr, action_mask, deterministic: bool = False) -> SACOutput:
        batch = torch.zeros(node_x.size(0), dtype=torch.long, device=node_x.device)
        logits, probs, _ = self.actor(node_x, edge_index, edge_attr, action_mask, batch)
        if deterministic:
            action = torch.argmax(probs).item()
            log_prob = torch.log(probs[action] + 1e-8)
        else:
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[action] + 1e-8)
        return SACOutput(action=action, log_prob=log_prob, probs=probs)

    def update(self, batch, weights=None, alpha_max: float = None):
        if isinstance(batch, list):
            samples = batch
        else:
            samples = [batch]
        if weights is None:
            weights = [1.0] * len(samples)

        critic_losses = []
        actor_losses = []
        alpha_losses = []
        td_errors = []

        for idx, sample in enumerate(samples):
            (
                node_x,
                edge_index,
                edge_attr,
                action_mask,
                batch_vec,
                action,
                reward,
                next_node_x,
                next_edge_attr,
                next_action_mask,
                next_batch_vec,
                done,
            ) = sample

            edge_batch = batch_vec[edge_index[0]]
            with torch.no_grad():
                _, next_probs, _ = self.actor(next_node_x, edge_index, next_edge_attr, next_action_mask, next_batch_vec)
                q1_next = self.target1(next_node_x, edge_index, next_edge_attr, next_batch_vec)
                q2_next = self.target2(next_node_x, edge_index, next_edge_attr, next_batch_vec)
                q_next = torch.min(q1_next, q2_next)
                v_next = scatter_sum(next_probs * (q_next - self.alpha * torch.log(next_probs + 1e-8)), edge_batch, dim=0)
                target = reward + (1.0 - done) * self.gamma * v_next

            q1_all = self.critic1(node_x, edge_index, edge_attr, batch_vec)
            q2_all = self.critic2(node_x, edge_index, edge_attr, batch_vec)
            q1 = q1_all[action]
            q2 = q2_all[action]

            td_error = (target - q1).detach()
            td_errors.extend(td_error.cpu().numpy().tolist())

            w = torch.as_tensor(weights[idx], device=reward.device, dtype=reward.dtype)
            critic_losses.append(w * (F.mse_loss(q1, target) + F.mse_loss(q2, target)))

            logits, probs, _ = self.actor(node_x, edge_index, edge_attr, action_mask, batch_vec)
            q_all = torch.min(q1_all, q2_all).detach()
            actor_terms = probs * (self.alpha * torch.log(probs + 1e-8) - q_all)
            actor_loss = scatter_sum(actor_terms, edge_batch, dim=0).mean()
            actor_losses.append(actor_loss)

            if self.target_entropy is None:
                valid = scatter_sum((action_mask > 0).float(), edge_batch, dim=0)
                target_entropy = (-0.6 * torch.log(valid + 1e-8)).mean()
            else:
                target_entropy = self.target_entropy
            log_probs = torch.log(probs + 1e-8).detach()
            alpha_term = scatter_sum(probs.detach() * (log_probs + target_entropy), edge_batch, dim=0)
            alpha_losses.append(-(self.log_alpha * alpha_term).mean())

        critic_loss = torch.stack(critic_losses).mean()
        actor_loss = torch.stack(actor_losses).mean()
        alpha_loss = torch.stack(alpha_losses).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.critic1.parameters()) + list(self.critic2.parameters()),
                max_norm=self.grad_clip,
            )
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
        self.actor_opt.step()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=self.grad_clip)
        self.alpha_opt.step()
        if alpha_max is not None:
            self.log_alpha.data.clamp_(max=float(np.log(alpha_max)))

        if self.share_critic_encoder:
            self._soft_update(self.critic_encoder, self.target_encoder)
            self._soft_update(self.critic1.edge_mlp, self.target1.edge_mlp)
            self._soft_update(self.critic2.edge_mlp, self.target2.edge_mlp)
        else:
            self._soft_update(self.critic1, self.target1)
            self._soft_update(self.critic2, self.target2)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "td_errors": td_errors,
        }

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target1": self.target1.state_dict(),
                "target2": self.target2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path: str, map_location: str = "cpu"):
        state = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.target1.load_state_dict(state["target1"])
        self.target2.load_state_dict(state["target2"])
        self.log_alpha = state["log_alpha"].to(map_location).requires_grad_()
        lr = self.alpha_opt.param_groups[0]["lr"]
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    def _soft_update(self, src, tgt):
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.target_tau) + p.data * self.target_tau)
