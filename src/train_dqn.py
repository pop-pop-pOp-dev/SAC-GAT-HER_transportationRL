from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.data.tntp_download import download_sioux_falls
from src.data.tntp_parser import load_graph_data
from src.env.repair_env import RepairEnv, EnvState
from src.models.gat_encoder import GATEncoder
from src.rl.rllib_models import _build_batched_graph


class QNetwork(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int, embed: int, num_layers: int, edge_index):
        super().__init__()
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.node_norm = nn.LayerNorm(node_in)
        self.edge_norm = nn.LayerNorm(edge_in)
        self.encoder = GATEncoder(node_in, hidden, embed, edge_dim=edge_in, num_layers=num_layers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed * 4 + edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_x: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
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
        q_vals = self.edge_mlp(edge_emb).squeeze(-1)
        q_vals = q_vals.view(batch_size, num_edges)
        return q_vals


def _stack_states(states: List[EnvState], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_x = torch.tensor(np.stack([s.node_features for s in states]), dtype=torch.float32, device=device)
    edge_attr = torch.tensor(np.stack([s.edge_features for s in states]), dtype=torch.float32, device=device)
    action_mask = torch.tensor(np.stack([s.action_mask for s in states]), dtype=torch.float32, device=device)
    return node_x, edge_attr, action_mask


def _select_action(q_values: torch.Tensor, action_mask: np.ndarray, epsilon: float) -> int:
    valid = np.where(action_mask > 0)[0]
    if valid.size == 0:
        return int(np.argmax(action_mask))
    if random.random() < epsilon:
        return int(np.random.choice(valid))
    q = q_values.detach().cpu().numpy()
    masked_q = np.full_like(q, -1e9)
    masked_q[valid] = q[valid]
    return int(masked_q.argmax())


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tgt, src in zip(target.parameters(), source.parameters()):
        tgt.data.copy_(tgt.data * (1.0 - tau) + src.data * tau)


def train(cfg):
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(cfg.get("device", "cuda"))
    data_paths = download_sioux_falls(cfg["data_dir"])
    graph = load_graph_data(data_paths["net_path"], data_paths["trips_path"])

    env = RepairEnv(
        graph,
        damaged_ratio=cfg["damaged_ratio"],
        assignment_iters=cfg["assignment_iters"],
        assignment_method=cfg.get("assignment_method", "msa"),
        use_cugraph=cfg.get("use_cugraph", False),
        use_torch=cfg.get("use_torch_bpr", False),
        device=str(device),
        sp_backend=cfg.get("sp_backend", "auto"),
        force_gpu_sp=cfg.get("force_gpu_sp", False),
        reward_mode=cfg.get("reward_mode", "log_delta"),
        reward_alpha=cfg.get("reward_alpha", 1.0),
        reward_beta=cfg.get("reward_beta", 10.0),
        reward_gamma=cfg.get("reward_gamma", 0.1),
        reward_clip=cfg.get("reward_clip", 0.0),
        capacity_damage=cfg.get("capacity_damage", 1e-3),
        unassigned_penalty=cfg.get("unassigned_penalty", 2e7),
        fixed_damage=cfg.get("fixed_damage", False),
        fixed_damage_seed=cfg.get("fixed_damage_seed"),
        seed=seed,
    )

    state = env.reset(damaged_ratio=cfg["damaged_ratio"])
    node_in = state.node_features.shape[1]
    edge_in = state.edge_features.shape[1]
    edge_index = state.edge_index

    q_net = QNetwork(
        node_in=node_in,
        edge_in=edge_in,
        hidden=int(cfg.get("hidden_dim", 256)),
        embed=int(cfg.get("embed_dim", 256)),
        num_layers=int(cfg.get("gat_layers", 3)),
        edge_index=edge_index,
    ).to(device)
    target_net = QNetwork(
        node_in=node_in,
        edge_in=edge_in,
        hidden=int(cfg.get("hidden_dim", 256)),
        embed=int(cfg.get("embed_dim", 256)),
        num_layers=int(cfg.get("gat_layers", 3)),
        edge_index=edge_index,
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=float(cfg.get("lr", 1e-4)))
    gamma = float(cfg.get("gamma", 0.99))
    tau = float(cfg.get("target_tau", 0.001))
    batch_size = int(cfg.get("batch_size", 256))
    buffer_size = int(cfg.get("buffer_size", 100000))
    batch_start = int(cfg.get("batch_start", 2000))
    update_every = int(cfg.get("update_every", 1))
    max_steps = int(cfg.get("max_steps", 0) or 0)
    reward_scale = float(cfg.get("reward_scale", 1.0))

    eps_start = float(cfg.get("eps_start", 1.0))
    eps_end = float(cfg.get("eps_end", 0.05))
    eps_decay = float(cfg.get("eps_decay", 0.995))
    epsilon = eps_start

    output_dir = cfg.get("output_dir", "./outputs")
    model_dir = cfg.get("model_dir", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    replay = deque(maxlen=buffer_size)
    best_tstt = float("inf")
    metrics = []

    episodes = int(cfg.get("episodes", 1000))
    for ep in range(episodes):
        state = env.reset(damaged_ratio=cfg["damaged_ratio"])
        done = False
        steps = 0
        total_reward = 0.0
        tstt_curve = []

        while not done:
            node_x = torch.tensor(state.node_features, dtype=torch.float32, device=device).unsqueeze(0)
            edge_attr = torch.tensor(state.edge_features, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = q_net(node_x, edge_attr)[0]
            action = _select_action(q_vals, state.action_mask, epsilon)

            next_state, reward, done, info = env.step(action)
            total_reward += reward * reward_scale
            tstt_curve.append(info.get("tstt", env.tstt))
            replay.append((state, action, reward, next_state, float(done)))
            state = next_state
            steps += 1

            if max_steps > 0 and steps >= max_steps and not done:
                done = True

            if len(replay) >= batch_start and steps % update_every == 0:
                batch = random.sample(replay, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                node_x, edge_attr, action_mask = _stack_states(list(states), device)
                next_node_x, next_edge_attr, next_action_mask = _stack_states(list(next_states), device)

                q_all = q_net(node_x, edge_attr)
                q_actions = q_all.gather(1, torch.tensor(actions, device=device).unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = target_net(next_node_x, next_edge_attr)
                    q_next = q_next.masked_fill(next_action_mask <= 0, -1e9)
                    q_next_max = q_next.max(dim=1)[0]
                    target = torch.tensor(rewards, device=device, dtype=torch.float32) + gamma * (
                        1.0 - torch.tensor(dones, device=device)
                    ) * q_next_max

                loss = F.mse_loss(q_actions, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), float(cfg.get("grad_clip", 1.0)))
                optimizer.step()
                _soft_update(target_net, q_net, tau)

        epsilon = max(eps_end, epsilon * eps_decay)
        tstt_last = float(tstt_curve[-1]) if tstt_curve else env.tstt
        tstt_mean = float(np.mean(tstt_curve)) if tstt_curve else env.tstt
        tstt_auc = float(np.trapz(tstt_curve)) if tstt_curve else env.tstt
        metrics.append(
            {
                "episode": ep,
                "reward": total_reward,
                "tstt_last": tstt_last,
                "tstt_mean": tstt_mean,
                "tstt_auc": tstt_auc,
                "epsilon": epsilon,
            }
        )

        print(
            f"[DQN] Ep {ep} | Reward {total_reward:.4f} | TSTT Last {tstt_last:.2f} "
            f"| AUC {tstt_auc:.2f} | eps {epsilon:.3f} | buf {len(replay)}",
            flush=True,
        )

        if tstt_last < best_tstt:
            best_tstt = tstt_last
            torch.save(q_net.state_dict(), os.path.join(model_dir, "model_best_eval.pt"))
        if (ep + 1) % 50 == 0:
            torch.save(q_net.state_dict(), os.path.join(model_dir, f"model_ep{ep+1}.pt"))

    torch.save(q_net.state_dict(), os.path.join(model_dir, "model_last.pt"))
    with open(os.path.join(output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sioux_falls_rllib_dqn.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
