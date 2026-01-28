from __future__ import annotations

import argparse
import os
import random
import time
import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch

from src.data.tntp_download import download_sioux_falls
from src.data.tntp_parser import load_graph_data
from src.env.repair_env import RepairEnv, EnvState
from src.rl.sac import DiscreteSAC, Actor


@dataclass
class ReplayBuffer:
    capacity: int
    alpha: float = 0.6
    beta: float = 0.4
    eps: float = 1e-6
    data: List = None
    tree: np.ndarray = None
    max_priority: float = 1.0
    ptr: int = 0
    size: int = 0

    def __post_init__(self):
        self.data = [None] * self.capacity
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)

    def _set_priority(self, idx: int, priority: float) -> None:
        tree_idx = idx + self.capacity
        delta = priority - self.tree[tree_idx]
        while tree_idx >= 1:
            self.tree[tree_idx] += delta
            tree_idx //= 2

    def add(self, item, priority: float = None):
        if priority is None:
            priority = self.max_priority
        priority = float(abs(priority) + self.eps)
        self.max_priority = max(self.max_priority, priority)
        p = priority**self.alpha
        self.data[self.ptr] = item
        self._set_priority(self.ptr, p)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        total = float(self.tree[1])
        if total <= 0 or self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            r = np.random.rand() * total
            idx = 1
            while idx < self.capacity:
                left = idx * 2
                if r <= self.tree[left]:
                    idx = left
                else:
                    r -= self.tree[left]
                    idx = left + 1
            data_idx = idx - self.capacity
            indices[i] = data_idx
            priorities[i] = self.tree[idx]
        probs = priorities / total
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max() if weights.max() > 0 else 1.0
        batch = [self.data[i] for i in indices]
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            priority = float(abs(err) + self.eps)
            self.max_priority = max(self.max_priority, priority)
            p = priority**self.alpha
            self._set_priority(int(i), p)


def to_torch(state, device):
    node_x = torch.tensor(state.node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(state.edge_index, dtype=torch.long, device=device)
    edge_attr = torch.tensor(state.edge_features, dtype=torch.float32, device=device)
    action_mask = torch.tensor(state.action_mask, dtype=torch.float32, device=device)
    return node_x, edge_index, edge_attr, action_mask


def state_to_data(state: EnvState) -> Data:
    return Data(
        x=torch.tensor(state.node_features, dtype=torch.float32),
        edge_index=torch.tensor(state.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(state.edge_features, dtype=torch.float32),
    )


def build_pyg_batch_from_data(data_list, action_masks, device):
    batch = Batch.from_data_list(data_list).to(device)
    action_mask = torch.cat(
        [torch.tensor(mask, dtype=torch.float32) for mask in action_masks],
        dim=0,
    ).to(device)
    return batch, action_mask


def build_pyg_batch(states, device):
    data_list = [state_to_data(s) for s in states]
    action_masks = [s.action_mask for s in states]
    return build_pyg_batch_from_data(data_list, action_masks, device)


def apply_goal(state: EnvState, goal: np.ndarray) -> EnvState:
    edge_features = state.edge_features.copy()
    edge_features[:, -1] = goal.astype(np.float32)
    return EnvState(
        node_features=state.node_features.copy(),
        edge_features=edge_features,
        edge_index=state.edge_index,
        action_mask=state.action_mask.copy(),
        log_tstt=state.log_tstt,
        goal_mask=goal.astype(np.float32),
    )


def rollout_worker(worker_id, cfg, weights_queue, out_queue, stop_event):
    device = torch.device("cpu")
    data_paths = download_sioux_falls(cfg["data_dir"])
    graph = load_graph_data(data_paths["net_path"], data_paths["trips_path"])
    env = RepairEnv(
        graph,
        damaged_ratio=cfg["damaged_ratio"],
        assignment_iters=cfg["assignment_iters"],
        assignment_method=cfg.get("assignment_method", "msa"),
        use_cugraph=cfg.get("worker_use_cugraph", False),
        use_torch=cfg.get("worker_use_torch_bpr", False),
        device=str(device),
        sp_backend=cfg.get("worker_sp_backend", "auto"),
        force_gpu_sp=cfg.get("worker_force_gpu_sp", False),
        reward_mode=cfg.get("reward_mode", "delta"),
        reward_alpha=cfg.get("reward_alpha", 1.0),
        reward_beta=cfg.get("reward_beta", 10.0),
        reward_gamma=cfg.get("reward_gamma", 0.1),
        reward_clip=cfg.get("reward_clip", 0.0),
        capacity_damage=cfg.get("capacity_damage", 1e-3),
        unassigned_penalty=cfg.get("unassigned_penalty", 2e7),
        debug_reward=False,
        seed=cfg["seed"] + 1000 + worker_id,
    )
    reward_scale = float(cfg.get("reward_scale", 1.0))
    rollout_steps = int(cfg.get("rollout_steps_per_worker", 5))
    state = env.reset(damaged_ratio=cfg["damaged_ratio"])
    actor = Actor(
        node_in=state.node_features.shape[1],
        edge_in=state.edge_features.shape[1],
        hidden=cfg["hidden_dim"],
        embed=cfg["embed_dim"],
        num_layers=cfg.get("gat_layers", 3),
    ).to(device)
    actor.eval()

    while not stop_event.is_set():
        # Refresh weights if provided.
        try:
            while True:
                weights = weights_queue.get_nowait()
                actor.load_state_dict(weights)
        except queue.Empty:
            pass

        for _ in range(rollout_steps):
            if stop_event.is_set():
                break
            node_x, edge_index, edge_attr, action_mask = to_torch(state, device)
            batch = torch.zeros(node_x.size(0), dtype=torch.long, device=device)
            with torch.no_grad():
                logits, probs, _ = actor(node_x, edge_index, edge_attr, action_mask, batch)
                action = torch.multinomial(probs, 1).item()
            prev_tstt = env.tstt
            next_state, reward, done, info = env.step(action)
            next_tstt = info.get("tstt", env.tstt)
            scaled_reward = reward * reward_scale
            out_queue.put(
                {
                    "type": "step",
                    "worker": worker_id,
                    "state": state,
                    "action": action,
                    "reward": scaled_reward,
                    "next_state": next_state,
                    "done": float(done),
                    "prev_tstt": prev_tstt,
                    "next_tstt": next_tstt,
                }
            )
            state = next_state
            if done:
                state = env.reset(damaged_ratio=cfg["damaged_ratio"])
                out_queue.put({"type": "episode_done", "worker": worker_id})
                break

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_override = os.environ.get("SEED_OVERRIDE")
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
        cfg["output_dir"] = os.path.join(cfg["output_dir"], f"seed_{cfg['seed']}")
        cfg["model_dir"] = cfg["output_dir"]
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
        reward_mode=cfg.get("reward_mode", "delta"),
        reward_alpha=cfg.get("reward_alpha", 1.0),
        reward_beta=cfg.get("reward_beta", 10.0),
        reward_gamma=cfg.get("reward_gamma", 0.1),
        reward_clip=cfg.get("reward_clip", 0.0),
        capacity_damage=cfg.get("capacity_damage", 1e-3),
        unassigned_penalty=cfg.get("unassigned_penalty", 2e7),
        debug_reward=cfg.get("debug_reward", False),
        debug_reward_every=cfg.get("debug_reward_every", 0),
        seed=cfg["seed"],
    )

    sample_state = env.get_state()
    node_in = sample_state.node_features.shape[1]
    edge_in = sample_state.edge_features.shape[1]

    agent = DiscreteSAC(
        node_in=node_in,
        edge_in=edge_in,
        hidden=cfg["hidden_dim"],
        embed=cfg["embed_dim"],
        num_layers=cfg.get("gat_layers", 3),
        lr=cfg["lr"],
        actor_lr=cfg.get("actor_lr"),
        critic_lr=cfg.get("critic_lr"),
        alpha_lr=cfg.get("alpha_lr"),
        grad_clip=cfg.get("grad_clip"),
        gamma=cfg["gamma"],
        target_tau=cfg["target_tau"],
        share_critic_encoder=cfg.get("share_critic_encoder", True),
        alpha_init=cfg.get("alpha_init", 0.1),
        target_entropy_ratio=cfg.get("target_entropy_ratio", 0.6),
    )
    agent.actor.to(device)
    agent.critic1.to(device)
    agent.critic2.to(device)
    agent.target1.to(device)
    agent.target2.to(device)

    replay = ReplayBuffer(
        capacity=cfg["buffer_size"],
        alpha=cfg.get("per_alpha", 0.6),
        beta=cfg.get("per_beta", 0.4),
        eps=cfg.get("per_eps", 1e-6),
    )

    metrics = []
    reward_hist = []
    tstt_mean_hist = []
    tstt_auc_hist = []
    critic_hist = []
    actor_hist = []
    alpha_loss_hist = []
    entropy_hist = []
    eval_tstt_hist = []
    os.makedirs(cfg["output_dir"], exist_ok=True)
    tb_dir = os.path.join(cfg["output_dir"], "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    model_dir = cfg.get("model_dir", cfg["output_dir"])
    os.makedirs(model_dir, exist_ok=True)
    fig_path = os.path.join(cfg["output_dir"], "train_curves.png")
    plot_every = int(cfg.get("plot_every", 1))
    reward_scale = float(cfg.get("reward_scale", 1.0))
    reward_mode = str(cfg.get("reward_mode", "delta"))
    reward_alpha = float(cfg.get("reward_alpha", 1.0))
    reward_beta = float(cfg.get("reward_beta", 10.0))
    reward_gamma = float(cfg.get("reward_gamma", 0.1))
    reward_clip = float(cfg.get("reward_clip", 0.0))
    max_steps = int(cfg.get("max_steps", 0))
    her_ratio = float(cfg.get("her_ratio", 0.0))
    update_every = int(cfg.get("update_every", 1))
    updates_per_step = int(cfg.get("updates_per_step", 1))
    alpha_max = cfg.get("alpha_max")
    save_best = bool(cfg.get("save_best", True))
    best_eval_tstt = float("inf")
    eval_every = int(cfg.get("eval_every", 50))
    eval_seeds = cfg.get("eval_seeds") or [1001, 1002, 1003, 1004, 1005]
    smooth_window = int(cfg.get("smooth_window", 10))
    plot_clip_percentile = float(cfg.get("plot_clip_percentile", 99.0))
    plot_clip_percentile_auc = float(cfg.get("plot_clip_percentile_auc", plot_clip_percentile))
    plot_clip_percentile_mean = float(cfg.get("plot_clip_percentile_mean", plot_clip_percentile))
    plot_tstt_log = bool(cfg.get("plot_tstt_log", True))

    def smooth_series(values, window):
        if window <= 1:
            return np.asarray(values, dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        window = min(int(window), int(arr.size))
        if window <= 1:
            return arr
        mask = ~np.isnan(arr)
        filled = np.where(mask, arr, 0.0)
        kernel = np.ones(window, dtype=np.float32)
        numer = np.convolve(filled, kernel, mode="same")
        denom = np.convolve(mask.astype(np.float32), kernel, mode="same")
        smoothed = np.divide(numer, np.maximum(denom, 1.0))
        smoothed[denom == 0] = np.nan
        return smoothed

    def clip_series(values, lower=1.0, upper=99.0):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        vals = arr[np.isfinite(arr)]
        if vals.size < 5:
            return arr
        lo = np.percentile(vals, lower)
        hi = np.percentile(vals, upper)
        if hi <= lo:
            return arr
        return np.clip(arr, lo, hi)

    def apply_ylim(ax, values, lower=1.0, upper=99.0):
        vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float32)
        if vals.size < 5:
            return
        lo = np.percentile(vals, lower)
        hi = np.percentile(vals, upper)
        if hi <= lo:
            return
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

    def record_episode(episode_idx, episode_reward, episode_tstt, last_losses, last_tstt, delta_tstt):
        tstt_mean = float(np.mean(episode_tstt)) if episode_tstt else last_tstt
        tstt_auc = float(np.trapz(episode_tstt)) if episode_tstt else last_tstt
        metrics.append(
            {
                "episode": episode_idx,
                "reward": episode_reward,
                "tstt_last": last_tstt,
                "tstt_mean": tstt_mean,
                "tstt_auc": tstt_auc,
            }
        )
        writer.add_scalar("train/delta_tstt", delta_tstt, episode_idx)
        reward_hist.append(episode_reward)
        tstt_mean_hist.append(tstt_mean)
        tstt_auc_hist.append(tstt_auc)
        critic_hist.append(last_losses.get("critic_loss") if last_losses else None)
        actor_hist.append(last_losses.get("actor_loss") if last_losses else None)
        alpha_loss_hist.append(last_losses.get("alpha_loss") if last_losses else None)
        entropy_hist.append(last_losses.get("policy_entropy") if last_losses else None)
        while len(eval_tstt_hist) < len(reward_hist):
            eval_tstt_hist.append(np.nan)
        writer.add_scalar("train/reward", episode_reward, episode_idx)
        writer.add_scalar("train/tstt_mean", tstt_mean, episode_idx)
        writer.add_scalar("train/tstt_auc", tstt_auc, episode_idx)
        if last_losses:
            writer.add_scalar("train/critic_loss", last_losses.get("critic_loss", 0.0), episode_idx)
            writer.add_scalar("train/actor_loss", last_losses.get("actor_loss", 0.0), episode_idx)
            writer.add_scalar("train/alpha", last_losses.get("alpha", 0.0), episode_idx)
            writer.add_scalar("train/alpha_loss", last_losses.get("alpha_loss", 0.0), episode_idx)
            writer.add_scalar("train/policy_entropy", last_losses.get("policy_entropy", 0.0), episode_idx)

        if plot_every > 0 and (episode_idx + 1) % plot_every == 0:
            fig, axes = plt.subplots(5, 2, figsize=(12, 20), sharex=True)
            x = np.arange(len(reward_hist))
            reward_plot = clip_series(reward_hist, upper=plot_clip_percentile)
            reward_smooth = smooth_series(reward_plot, smooth_window)
            axes[0, 0].plot(x, reward_plot, color="#1f77b4", label="Reward")
            axes[0, 0].plot(x, reward_smooth, color="#00e5ff", linestyle="--", label="Reward (smoothed)")
            axes[0, 0].set_title("Reward")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Scaled Reward")
            axes[0, 0].legend()
            apply_ylim(axes[0, 0], reward_plot)

            tstt_mean_raw = tstt_mean_hist
            if plot_tstt_log:
                tstt_mean_raw = np.log10(np.maximum(np.asarray(tstt_mean_raw, dtype=np.float32), 1e-6))
            tstt_mean_plot = clip_series(tstt_mean_raw, upper=plot_clip_percentile_mean)
            tstt_mean_smooth = smooth_series(tstt_mean_plot, smooth_window)
            axes[0, 1].plot(x, tstt_mean_plot, color="#2ca02c", label="TSTT Mean")
            axes[0, 1].plot(x, tstt_mean_smooth, color="#00ff4f", linestyle="--", label="TSTT Mean (smoothed)")
            axes[0, 1].set_title("TSTT Mean")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("log10(TSTT)" if plot_tstt_log else "TSTT")
            axes[0, 1].legend()
            apply_ylim(axes[0, 1], tstt_mean_plot)

            tstt_auc_raw = tstt_auc_hist
            if plot_tstt_log:
                tstt_auc_raw = np.log10(np.maximum(np.asarray(tstt_auc_raw, dtype=np.float32), 1e-6))
            tstt_auc_plot = clip_series(tstt_auc_raw, upper=plot_clip_percentile_auc)
            tstt_auc_smooth = smooth_series(tstt_auc_plot, smooth_window)
            axes[1, 0].plot(x, tstt_auc_plot, color="#9467bd", label="TSTT AUC")
            axes[1, 0].plot(x, tstt_auc_smooth, color="#ff00ff", linestyle="--", label="TSTT AUC (smoothed)")
            axes[1, 0].set_title("TSTT AUC")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("log10(AUC)" if plot_tstt_log else "AUC")
            axes[1, 0].legend()
            apply_ylim(axes[1, 0], tstt_auc_plot)

            critic_vals = [v if v is not None else np.nan for v in critic_hist]
            critic_smooth = smooth_series(critic_vals, smooth_window)
            axes[1, 1].plot(x, critic_vals, color="#d62728", label="Critic Loss")
            axes[1, 1].plot(x, critic_smooth, color="#ff004c", linestyle="--", label="Critic Loss (smoothed)")
            axes[1, 1].set_title("Critic Loss")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].legend()
            apply_ylim(axes[1, 1], critic_vals)

            actor_vals = [v if v is not None else np.nan for v in actor_hist]
            actor_smooth = smooth_series(actor_vals, smooth_window)
            axes[2, 0].plot(x, actor_vals, color="#ff7f0e", label="Actor Loss")
            axes[2, 0].plot(x, actor_smooth, color="#ff8c00", linestyle="--", label="Actor Loss (smoothed)")
            axes[2, 0].set_title("Actor Loss")
            axes[2, 0].set_xlabel("Episode")
            axes[2, 0].set_ylabel("Loss")
            axes[2, 0].legend()
            apply_ylim(axes[2, 0], actor_vals)

            tstt_last_vals = [m["tstt_last"] for m in metrics]
            tstt_last_plot = clip_series(tstt_last_vals, upper=plot_clip_percentile)
            tstt_last_smooth = smooth_series(tstt_last_plot, smooth_window)
            axes[2, 1].plot(x, tstt_last_plot, color="#8c564b", label="TSTT Last")
            axes[2, 1].plot(x, tstt_last_smooth, color="#ff00a8", linestyle="--", label="TSTT Last (smoothed)")
            axes[2, 1].set_title("TSTT Last (Episode End)")
            axes[2, 1].set_xlabel("Episode")
            axes[2, 1].set_ylabel("TSTT")
            axes[2, 1].legend()
            apply_ylim(axes[2, 1], tstt_last_plot)

            alpha_loss_vals = [v if v is not None else np.nan for v in alpha_loss_hist]
            alpha_loss_smooth = smooth_series(alpha_loss_vals, smooth_window)
            axes[3, 0].plot(x, alpha_loss_vals, color="#17becf", label="Alpha Loss")
            axes[3, 0].plot(x, alpha_loss_smooth, color="#00f5ff", linestyle="--", label="Alpha Loss (smoothed)")
            axes[3, 0].set_title("Alpha Loss")
            axes[3, 0].set_xlabel("Episode")
            axes[3, 0].set_ylabel("Loss")
            axes[3, 0].legend()
            apply_ylim(axes[3, 0], alpha_loss_vals)

            entropy_vals = [v if v is not None else np.nan for v in entropy_hist]
            entropy_smooth = smooth_series(entropy_vals, smooth_window)
            axes[3, 1].plot(x, entropy_vals, color="#7f7f7f", label="Policy Entropy")
            axes[3, 1].plot(x, entropy_smooth, color="#b6ff00", linestyle="--", label="Policy Entropy (smoothed)")
            axes[3, 1].set_title("Policy Entropy")
            axes[3, 1].set_xlabel("Episode")
            axes[3, 1].set_ylabel("Entropy")
            axes[3, 1].legend()
            apply_ylim(axes[3, 1], entropy_vals)

            eval_tstt_vals = [v if v is not None and np.isfinite(v) else np.nan for v in eval_tstt_hist]
            axes[4, 0].plot(
                x,
                eval_tstt_vals,
                color="#1f9bff",
                marker="o",
                markersize=3,
                linestyle="-",
                label="Eval TSTT",
            )
            axes[4, 0].set_title("Eval TSTT")
            axes[4, 0].set_xlabel("Episode")
            axes[4, 0].set_ylabel("TSTT")
            axes[4, 0].legend()
            apply_ylim(axes[4, 0], eval_tstt_vals)

            axes[4, 1].axis("off")

            for ax in axes.ravel():
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)

    def run_eval(episode_idx):
        nonlocal best_eval_tstt
        agent.actor.eval()
        eval_rewards = []
        eval_tstt = []
        eval_auc = []
        for seed in eval_seeds:
            eval_env = RepairEnv(
                graph,
                damaged_ratio=cfg["damaged_ratio"],
                assignment_iters=cfg["assignment_iters"],
                assignment_method=cfg.get("assignment_method", "msa"),
                use_cugraph=cfg.get("use_cugraph", False),
                use_torch=cfg.get("use_torch_bpr", False),
                device=str(device),
                sp_backend=cfg.get("sp_backend", "auto"),
                force_gpu_sp=cfg.get("force_gpu_sp", False),
                reward_mode=cfg.get("reward_mode", "delta"),
                reward_alpha=cfg.get("reward_alpha", 1.0),
                reward_beta=cfg.get("reward_beta", 10.0),
                reward_gamma=cfg.get("reward_gamma", 0.1),
                reward_clip=cfg.get("reward_clip", 0.0),
                capacity_damage=cfg.get("capacity_damage", 1e-3),
                unassigned_penalty=cfg.get("unassigned_penalty", 2e7),
                seed=seed,
            )
            state = eval_env.reset(damaged_ratio=cfg["damaged_ratio"])
            done = False
            steps = 0
            total_reward = 0.0
            tstt_curve = []
            while not done:
                node_x, edge_index, edge_attr, action_mask = to_torch(state, device)
                with torch.no_grad():
                    out = agent.select_action(node_x, edge_index, edge_attr, action_mask, deterministic=True)
                state, reward, done, info = eval_env.step(out.action)
                total_reward += reward * reward_scale
                tstt_curve.append(info.get("tstt", eval_env.tstt))
                steps += 1
                if max_steps > 0 and steps >= max_steps and not done:
                    break
            eval_rewards.append(total_reward)
            eval_tstt.append(float(tstt_curve[-1]) if tstt_curve else eval_env.tstt)
            eval_auc.append(float(np.trapz(tstt_curve)) if tstt_curve else eval_env.tstt)

        avg_reward = float(np.mean(eval_rewards))
        avg_tstt = float(np.mean(eval_tstt))
        avg_auc = float(np.mean(eval_auc))
        while len(eval_tstt_hist) < episode_idx:
            eval_tstt_hist.append(np.nan)
        if len(eval_tstt_hist) == episode_idx:
            eval_tstt_hist.append(avg_tstt)
        else:
            eval_tstt_hist[episode_idx] = avg_tstt
        writer.add_scalar("eval/avg_reward", avg_reward, episode_idx)
        writer.add_scalar("eval/avg_tstt", avg_tstt, episode_idx)
        writer.add_scalar("eval/avg_auc", avg_auc, episode_idx)

        if save_best and avg_tstt < best_eval_tstt:
            best_eval_tstt = avg_tstt
            best_path = os.path.join(model_dir, "model_best_eval.pt")
            agent.save(best_path)

        agent.actor.train()
    def resolve_worker_settings(cfg):
        cpu_count = os.cpu_count() or 1
        raw_workers = cfg.get("num_workers", None)
        if raw_workers is None:
            num_workers = max(1, min(16, cpu_count - 2))
        elif isinstance(raw_workers, str) and raw_workers.lower() == "auto":
            num_workers = max(1, min(16, cpu_count - 2))
        else:
            num_workers = int(raw_workers)
            if num_workers < 0:
                num_workers = max(1, min(16, cpu_count - 2))

        raw_steps = cfg.get("rollout_steps_per_worker", 0)
        if isinstance(raw_steps, str) and raw_steps.lower() == "auto":
            rollout_steps = max(2, min(20, 256 // max(1, num_workers)))
        else:
            rollout_steps = int(raw_steps) if raw_steps else 0
            if rollout_steps <= 0:
                rollout_steps = max(2, min(20, 256 // max(1, num_workers)))
        return num_workers, rollout_steps

    num_workers, rollout_steps = resolve_worker_settings(cfg)
    def _color(txt: str, code: str) -> str:
        return f"\033[{code}m{txt}\033[0m"

    torch_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if torch_cuda else "CPU"
    print(
        _color(
            f"[accel] device={device} torch_cuda={torch_cuda} gpu={device_name}",
            "92" if torch_cuda else "91",
        )
    )
    print(
        _color(
            f"[accel] sp_backend={cfg.get('sp_backend', 'auto')} force_gpu_sp={cfg.get('force_gpu_sp', False)} use_torch_bpr={cfg.get('use_torch_bpr', False)}",
            "96",
        )
    )
    if num_workers > 0:
        print(
            _color(
                f"[accel] parallel rollout=ON workers={num_workers} rollout_steps={rollout_steps}",
                "92",
            )
        )
        print(
            _color(
                f"[accel] worker backend=CPU sp_backend={cfg.get('worker_sp_backend', 'auto')} use_torch_bpr={cfg.get('worker_use_torch_bpr', False)}",
                "93",
            )
        )
    else:
        print(
            _color(
                f"[accel] parallel rollout=OFF workers={num_workers}",
                "93",
            )
        )
    print(
        _color(
            f"[accel] batch_size={cfg.get('batch_size')} update_every={cfg.get('update_every')} updates_per_step={cfg.get('updates_per_step')}",
            "96",
        )
    )
    if num_workers > 0:
        ctx = mp.get_context("spawn")
        queue_size = int(cfg.get("queue_size", 10000))
        weights_sync_every = int(cfg.get("weights_sync_every", 200))
        out_queue = ctx.Queue(maxsize=queue_size)
        stop_event = ctx.Event()
        weights_queues = []
        workers = []
        for i in range(num_workers):
            wq = ctx.Queue(maxsize=1)
            wq.put(agent.actor.state_dict())
            cfg["rollout_steps_per_worker"] = rollout_steps
            p = ctx.Process(target=rollout_worker, args=(i, cfg, wq, out_queue, stop_event))
            p.daemon = True
            p.start()
            weights_queues.append(wq)
            workers.append(p)

        worker_rewards = [0.0 for _ in range(num_workers)]
        worker_tstt = [[] for _ in range(num_workers)]
        worker_last_delta = [0.0 for _ in range(num_workers)]
        episodes_done = 0
        global_steps = 0
        last_losses = {}

        pbar = tqdm(total=cfg["episodes"])
        while episodes_done < cfg["episodes"]:
            try:
                msg = out_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get("type") == "step":
                wid = msg["worker"]
                state = msg["state"]
                next_state = msg["next_state"]
                prev_tstt = msg["prev_tstt"]
                next_tstt = msg["next_tstt"]
                delta_tstt = prev_tstt - next_tstt
                worker_last_delta[wid] = delta_tstt
                worker_rewards[wid] += msg["reward"]
                worker_tstt[wid].append(next_tstt)

                state_data = state_to_data(state)
                next_state_data = state_to_data(next_state)
                replay.add(
                    (
                        state,
                        msg["action"],
                        msg["reward"],
                        next_state,
                        msg["done"],
                        state.goal_mask,
                        prev_tstt,
                        next_tstt,
                        state_data,
                        next_state_data,
                    )
                )
                global_steps += 1

                if replay.size > cfg["batch_start"] and (global_steps % update_every == 0):
                    for _ in range(updates_per_step):
                        bs = min(cfg["batch_size"], replay.size)
                        batch_items, indices, weights = replay.sample(bs)
                        if any(item is None for item in batch_items):
                            continue
                        states = []
                        next_states = []
                        actions = []
                        rewards = []
                        dones = []
                        data_list = []
                        next_data_list = []
                        for s, a, r, s2, d, goal, prev_t, next_t, s_data, s2_data in batch_items:
                            if her_ratio > 0 and np.random.rand() < her_ratio:
                                achieved_goal = (1.0 - s2.action_mask).astype(np.float32)
                                goal = achieved_goal
                                r = env.compute_reward_with_goal(
                                    prev_t,
                                    next_t,
                                    goal,
                                    s2.action_mask,
                                    alpha=reward_alpha,
                                    beta=reward_beta,
                                    gamma=reward_gamma,
                                    mode=reward_mode,
                                    clip=reward_clip,
                                ) * reward_scale
                                d = float(env.is_goal_complete(goal, s2.action_mask))
                                s = apply_goal(s, goal)
                                s2 = apply_goal(s2, goal)
                                s_data = state_to_data(s)
                                s2_data = state_to_data(s2)
                            states.append(s)
                            next_states.append(s2)
                            data_list.append(s_data)
                            next_data_list.append(s2_data)
                            actions.append(a)
                            rewards.append(r)
                            dones.append(d)
                        batch_state, action_mask = build_pyg_batch_from_data(
                            data_list,
                            [s.action_mask for s in states],
                            device,
                        )
                        batch_next_state, next_action_mask = build_pyg_batch_from_data(
                            next_data_list,
                            [s.action_mask for s in next_states],
                            device,
                        )
                        edge_batch = batch_state.batch[batch_state.edge_index[0]]
                        global_actions = []
                        for i, a in enumerate(actions):
                            idxs = (edge_batch == i).nonzero(as_tuple=False).squeeze(-1)
                            global_actions.append(int(idxs[a]))
                        batch = (
                            batch_state.x,
                            batch_state.edge_index,
                            batch_state.edge_attr,
                            action_mask,
                            batch_state.batch,
                            torch.tensor(global_actions, dtype=torch.long, device=device),
                            torch.tensor(rewards, dtype=torch.float32, device=device),
                            batch_next_state.x,
                            batch_next_state.edge_attr,
                            next_action_mask,
                            batch_next_state.batch,
                            torch.tensor(dones, dtype=torch.float32, device=device),
                        )
                        last_losses = agent.update(batch, weights=weights, alpha_max=alpha_max)
                        replay.update_priorities(indices, last_losses.get("td_errors", []))

                if weights_sync_every > 0 and (global_steps % weights_sync_every == 0):
                    weights = {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()}
                    for wq in weights_queues:
                        try:
                            while True:
                                wq.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            wq.put_nowait(weights)
                        except queue.Full:
                            pass

                if msg["done"]:
                    record_episode(
                        episodes_done,
                        worker_rewards[wid],
                        worker_tstt[wid],
                        last_losses,
                        next_tstt,
                        worker_last_delta[wid],
                    )
                    worker_rewards[wid] = 0.0
                    worker_tstt[wid] = []
                    episodes_done += 1
                    pbar.update(1)
                    if eval_every > 0 and episodes_done % eval_every == 0:
                        run_eval(episodes_done)

        pbar.close()
        stop_event.set()
        for p in workers:
            p.join(timeout=2.0)
    else:
        for episode in tqdm(range(cfg["episodes"])):
            state = env.reset(damaged_ratio=cfg["damaged_ratio"])
            done = False
            episode_reward = 0.0
            steps = 0
            episode_tstt = []
            last_losses = {}
            while not done:
                node_x, edge_index, edge_attr, action_mask = to_torch(state, device)
                out = agent.select_action(node_x, edge_index, edge_attr, action_mask, deterministic=False)
                prev_tstt = env.tstt
                next_state, reward, done, info = env.step(out.action)
                next_tstt = info.get("tstt", env.tstt)
                delta_tstt = prev_tstt - next_tstt
                episode_tstt.append(next_tstt)
                scaled_reward = reward * reward_scale
                state_data = state_to_data(state)
                next_state_data = state_to_data(next_state)
                replay.add(
                    (
                        state,
                        out.action,
                        scaled_reward,
                        next_state,
                        float(done),
                        state.goal_mask,
                        prev_tstt,
                        next_tstt,
                        state_data,
                        next_state_data,
                    )
                )
                state = next_state
                episode_reward += scaled_reward
                steps += 1
                if max_steps > 0 and steps >= max_steps and not done:
                    # Truncated episode: do not mark terminal for critic targets.
                    break

                if replay.size > cfg["batch_start"] and (steps % update_every == 0):
                    for _ in range(updates_per_step):
                        bs = min(cfg["batch_size"], replay.size)
                        batch_items, indices, weights = replay.sample(bs)
                        if any(item is None for item in batch_items):
                            continue
                        states = []
                        next_states = []
                        actions = []
                        rewards = []
                        dones = []
                        data_list = []
                        next_data_list = []
                        for s, a, r, s2, d, goal, prev_t, next_t, s_data, s2_data in batch_items:
                            if her_ratio > 0 and np.random.rand() < her_ratio:
                                achieved_goal = (1.0 - s2.action_mask).astype(np.float32)
                                goal = achieved_goal
                                r = env.compute_reward_with_goal(
                                    prev_t,
                                    next_t,
                                    goal,
                                    s2.action_mask,
                                    alpha=reward_alpha,
                                    beta=reward_beta,
                                    gamma=reward_gamma,
                                    mode=reward_mode,
                                    clip=reward_clip,
                                ) * reward_scale
                                d = float(env.is_goal_complete(goal, s2.action_mask))
                                s = apply_goal(s, goal)
                                s2 = apply_goal(s2, goal)
                                s_data = state_to_data(s)
                                s2_data = state_to_data(s2)
                            states.append(s)
                            next_states.append(s2)
                            data_list.append(s_data)
                            next_data_list.append(s2_data)
                            actions.append(a)
                            rewards.append(r)
                            dones.append(d)
                        batch_state, action_mask = build_pyg_batch_from_data(
                            data_list,
                            [s.action_mask for s in states],
                            device,
                        )
                        batch_next_state, next_action_mask = build_pyg_batch_from_data(
                            next_data_list,
                            [s.action_mask for s in next_states],
                            device,
                        )
                        edge_batch = batch_state.batch[batch_state.edge_index[0]]
                        global_actions = []
                        for i, a in enumerate(actions):
                            idxs = (edge_batch == i).nonzero(as_tuple=False).squeeze(-1)
                            global_actions.append(int(idxs[a]))
                        batch = (
                            batch_state.x,
                            batch_state.edge_index,
                            batch_state.edge_attr,
                            action_mask,
                            batch_state.batch,
                            torch.tensor(global_actions, dtype=torch.long, device=device),
                            torch.tensor(rewards, dtype=torch.float32, device=device),
                            batch_next_state.x,
                            batch_next_state.edge_attr,
                            next_action_mask,
                            batch_next_state.batch,
                            torch.tensor(dones, dtype=torch.float32, device=device),
                        )
                        last_losses = agent.update(batch, weights=weights, alpha_max=alpha_max)
                        replay.update_priorities(indices, last_losses.get("td_errors", []))

            record_episode(episode, episode_reward, episode_tstt, last_losses, env.tstt, delta_tstt)
            if eval_every > 0 and (episode + 1) % eval_every == 0:
                run_eval(episode + 1)

    out_path = os.path.join(cfg["output_dir"], "train_metrics.npy")
    np.save(out_path, metrics)
    if plot_every <= 0:
        fig, axes = plt.subplots(5, 2, figsize=(12, 20), sharex=True)
        x = np.arange(len(reward_hist))
        reward_smooth = smooth_series(reward_hist, smooth_window)
        axes[0, 0].plot(x, reward_hist, color="#1f77b4", label="Reward")
        axes[0, 0].plot(x, reward_smooth, color="#00e5ff", linestyle="--", label="Reward (smoothed)")
        axes[0, 0].set_title("Reward")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Scaled Reward")
        axes[0, 0].legend()
        apply_ylim(axes[0, 0], reward_hist)

        tstt_mean_raw = tstt_mean_hist
        if plot_tstt_log:
            tstt_mean_raw = np.log10(np.maximum(np.asarray(tstt_mean_raw, dtype=np.float32), 1e-6))
        tstt_mean_smooth = smooth_series(tstt_mean_raw, smooth_window)
        axes[0, 1].plot(x, tstt_mean_raw, color="#2ca02c", label="TSTT Mean")
        axes[0, 1].plot(x, tstt_mean_smooth, color="#00ff4f", linestyle="--", label="TSTT Mean (smoothed)")
        axes[0, 1].set_title("TSTT Mean")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("log10(TSTT)" if plot_tstt_log else "TSTT")
        axes[0, 1].legend()
        apply_ylim(axes[0, 1], tstt_mean_raw)

        tstt_auc_raw = tstt_auc_hist
        if plot_tstt_log:
            tstt_auc_raw = np.log10(np.maximum(np.asarray(tstt_auc_raw, dtype=np.float32), 1e-6))
        tstt_auc_smooth = smooth_series(tstt_auc_raw, smooth_window)
        axes[1, 0].plot(x, tstt_auc_raw, color="#9467bd", label="TSTT AUC")
        axes[1, 0].plot(x, tstt_auc_smooth, color="#ff00ff", linestyle="--", label="TSTT AUC (smoothed)")
        axes[1, 0].set_title("TSTT AUC")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("log10(AUC)" if plot_tstt_log else "AUC")
        axes[1, 0].legend()
        apply_ylim(axes[1, 0], tstt_auc_raw)

        critic_vals = [v if v is not None else np.nan for v in critic_hist]
        critic_smooth = smooth_series(critic_vals, smooth_window)
        axes[1, 1].plot(x, critic_vals, color="#d62728", label="Critic Loss")
        axes[1, 1].plot(x, critic_smooth, color="#ff004c", linestyle="--", label="Critic Loss (smoothed)")
        axes[1, 1].set_title("Critic Loss")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].legend()
        apply_ylim(axes[1, 1], critic_vals)

        actor_vals = [v if v is not None else np.nan for v in actor_hist]
        actor_smooth = smooth_series(actor_vals, smooth_window)
        axes[2, 0].plot(x, actor_vals, color="#ff7f0e", label="Actor Loss")
        axes[2, 0].plot(x, actor_smooth, color="#ff8c00", linestyle="--", label="Actor Loss (smoothed)")
        axes[2, 0].set_title("Actor Loss")
        axes[2, 0].set_xlabel("Episode")
        axes[2, 0].set_ylabel("Loss")
        axes[2, 0].legend()
        apply_ylim(axes[2, 0], actor_vals)

        tstt_last_vals = [m["tstt_last"] for m in metrics]
        tstt_last_plot = clip_series(tstt_last_vals, upper=plot_clip_percentile)
        tstt_last_smooth = smooth_series(tstt_last_plot, smooth_window)
        axes[2, 1].plot(x, tstt_last_plot, color="#8c564b", label="TSTT Last")
        axes[2, 1].plot(x, tstt_last_smooth, color="#ff00a8", linestyle="--", label="TSTT Last (smoothed)")
        axes[2, 1].set_title("TSTT Last (Episode End)")
        axes[2, 1].set_xlabel("Episode")
        axes[2, 1].set_ylabel("TSTT")
        axes[2, 1].legend()
        apply_ylim(axes[2, 1], tstt_last_plot)

        alpha_loss_vals = [v if v is not None else np.nan for v in alpha_loss_hist]
        alpha_loss_smooth = smooth_series(alpha_loss_vals, smooth_window)
        axes[3, 0].plot(x, alpha_loss_vals, color="#17becf", label="Alpha Loss")
        axes[3, 0].plot(x, alpha_loss_smooth, color="#00f5ff", linestyle="--", label="Alpha Loss (smoothed)")
        axes[3, 0].set_title("Alpha Loss")
        axes[3, 0].set_xlabel("Episode")
        axes[3, 0].set_ylabel("Loss")
        axes[3, 0].legend()
        apply_ylim(axes[3, 0], alpha_loss_vals)

        entropy_vals = [v if v is not None else np.nan for v in entropy_hist]
        entropy_smooth = smooth_series(entropy_vals, smooth_window)
        axes[3, 1].plot(x, entropy_vals, color="#7f7f7f", label="Policy Entropy")
        axes[3, 1].plot(x, entropy_smooth, color="#b6ff00", linestyle="--", label="Policy Entropy (smoothed)")
        axes[3, 1].set_title("Policy Entropy")
        axes[3, 1].set_xlabel("Episode")
        axes[3, 1].set_ylabel("Entropy")
        axes[3, 1].legend()
        apply_ylim(axes[3, 1], entropy_vals)

        eval_tstt_vals = [v if v is not None and np.isfinite(v) else np.nan for v in eval_tstt_hist]
        axes[4, 0].plot(
            x,
            eval_tstt_vals,
            color="#1f9bff",
            marker="o",
            markersize=3,
            linestyle="-",
            label="Eval TSTT",
        )
        axes[4, 0].set_title("Eval TSTT")
        axes[4, 0].set_xlabel("Episode")
        axes[4, 0].set_ylabel("TSTT")
        axes[4, 0].legend()
        apply_ylim(axes[4, 0], eval_tstt_vals)

        axes[4, 1].axis("off")

        for ax in axes.ravel():
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
    model_dir = cfg.get("model_dir", cfg["output_dir"])
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_last.pt")
    agent.save(model_path)
    writer.close()
    print(f"Saved metrics to {out_path}")
    print(f"Saved curves to {fig_path}")
    print(f"Saved model to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sioux_falls.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train(cfg)


if __name__ == "__main__":
    main()
