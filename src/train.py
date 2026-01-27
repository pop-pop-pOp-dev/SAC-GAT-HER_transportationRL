from __future__ import annotations

import argparse
import os
import random
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
from src.rl.sac import DiscreteSAC


@dataclass
class ReplayBuffer:
    capacity: int
    alpha: float = 0.6
    beta: float = 0.4
    eps: float = 1e-6
    data: List = None
    priorities: List = None
    ptr: int = 0
    size: int = 0

    def __post_init__(self):
        self.data = [None] * self.capacity
        self.priorities = [0.0] * self.capacity

    def add(self, item, priority: float = None):
        if priority is None:
            priority = max(self.priorities) if self.size > 0 else 1.0
        self.data[self.ptr] = item
        self.priorities[self.ptr] = priority
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        probs = np.array(self.priorities[: self.size], dtype=np.float32) ** self.alpha
        probs /= probs.sum()
        replace = batch_size > self.size
        indices = np.random.choice(self.size, size=batch_size, replace=replace, p=probs)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        batch = [self.data[i] for i in indices]
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            self.priorities[i] = float(abs(err) + self.eps)


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

    agent = DiscreteSAC(
        node_in=3,
        edge_in=6,
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
    tb_dir = os.path.join(cfg["output_dir"], "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    os.makedirs(cfg["output_dir"], exist_ok=True)
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
    alpha_max = cfg.get("alpha_max")
    save_best = bool(cfg.get("save_best", True))
    best_auc = float("inf")
    smooth_window = int(cfg.get("smooth_window", 10))

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

            if replay.size > cfg["batch_start"]:
                bs = min(cfg["batch_size"], replay.size)
                batch_items, indices, weights = replay.sample(bs)
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

        tstt_mean = float(np.mean(episode_tstt)) if episode_tstt else env.tstt
        tstt_auc = float(np.trapezoid(episode_tstt)) if episode_tstt else env.tstt
        metrics.append(
            {
                "episode": episode,
                "reward": episode_reward,
                "tstt_last": env.tstt,
                "tstt_mean": tstt_mean,
                "tstt_auc": tstt_auc,
            }
        )
        if save_best and tstt_auc < best_auc:
            best_auc = tstt_auc
            best_path = os.path.join(model_dir, "model_best.pt")
            agent.save(best_path)
        writer.add_scalar("train/delta_tstt", delta_tstt, episode)
        reward_hist.append(episode_reward)
        tstt_mean_hist.append(tstt_mean)
        tstt_auc_hist.append(tstt_auc)
        critic_hist.append(last_losses.get("critic_loss") if last_losses else None)
        actor_hist.append(last_losses.get("actor_loss") if last_losses else None)
        writer.add_scalar("train/reward", episode_reward, episode)
        writer.add_scalar("train/tstt_mean", tstt_mean, episode)
        writer.add_scalar("train/tstt_auc", tstt_auc, episode)
        if last_losses:
            writer.add_scalar("train/critic_loss", last_losses.get("critic_loss", 0.0), episode)
            writer.add_scalar("train/actor_loss", last_losses.get("actor_loss", 0.0), episode)
            writer.add_scalar("train/alpha", last_losses.get("alpha", 0.0), episode)
        if plot_every > 0 and (episode + 1) % plot_every == 0:
            fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
            x = np.arange(len(reward_hist))
            reward_smooth = smooth_series(reward_hist, smooth_window)
            axes[0, 0].plot(x, reward_hist, color="#1f77b4", label="Reward")
            axes[0, 0].plot(x, reward_smooth, color="#1f77b4", linestyle="--", label="Reward (smoothed)")
            axes[0, 0].set_title("Reward")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Scaled Reward")
            axes[0, 0].legend()

            tstt_mean_smooth = smooth_series(tstt_mean_hist, smooth_window)
            axes[0, 1].plot(x, tstt_mean_hist, color="#2ca02c", label="TSTT Mean")
            axes[0, 1].plot(x, tstt_mean_smooth, color="#2ca02c", linestyle="--", label="TSTT Mean (smoothed)")
            axes[0, 1].set_title("TSTT Mean")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("TSTT")
            axes[0, 1].legend()

            tstt_auc_smooth = smooth_series(tstt_auc_hist, smooth_window)
            axes[1, 0].plot(x, tstt_auc_hist, color="#9467bd", label="TSTT AUC")
            axes[1, 0].plot(x, tstt_auc_smooth, color="#9467bd", linestyle="--", label="TSTT AUC (smoothed)")
            axes[1, 0].set_title("TSTT AUC")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("AUC")
            axes[1, 0].legend()

            critic_vals = [v if v is not None else np.nan for v in critic_hist]
            critic_smooth = smooth_series(critic_vals, smooth_window)
            axes[1, 1].plot(x, critic_vals, color="#d62728", label="Critic Loss")
            axes[1, 1].plot(x, critic_smooth, color="#d62728", linestyle="--", label="Critic Loss (smoothed)")
            axes[1, 1].set_title("Critic Loss")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].legend()

            actor_vals = [v if v is not None else np.nan for v in actor_hist]
            actor_smooth = smooth_series(actor_vals, smooth_window)
            axes[2, 0].plot(x, actor_vals, color="#ff7f0e", label="Actor Loss")
            axes[2, 0].plot(x, actor_smooth, color="#ff7f0e", linestyle="--", label="Actor Loss (smoothed)")
            axes[2, 0].set_title("Actor Loss")
            axes[2, 0].set_xlabel("Episode")
            axes[2, 0].set_ylabel("Loss")
            axes[2, 0].legend()

            tstt_last_vals = [m["tstt_last"] for m in metrics]
            tstt_last_smooth = smooth_series(tstt_last_vals, smooth_window)
            axes[2, 1].plot(x, tstt_last_vals, color="#8c564b", label="TSTT Last")
            axes[2, 1].plot(x, tstt_last_smooth, color="#8c564b", linestyle="--", label="TSTT Last (smoothed)")
            axes[2, 1].set_title("TSTT Last (Episode End)")
            axes[2, 1].set_xlabel("Episode")
            axes[2, 1].set_ylabel("TSTT")
            axes[2, 1].legend()

            for ax in axes.ravel():
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)

    out_path = os.path.join(cfg["output_dir"], "train_metrics.npy")
    np.save(out_path, metrics)
    if plot_every <= 0:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        x = np.arange(len(reward_hist))
        reward_smooth = smooth_series(reward_hist, smooth_window)
        axes[0, 0].plot(x, reward_hist, color="#1f77b4", label="Reward")
        axes[0, 0].plot(x, reward_smooth, color="#1f77b4", linestyle="--", label="Reward (smoothed)")
        axes[0, 0].set_title("Reward")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Scaled Reward")
        axes[0, 0].legend()

        tstt_mean_smooth = smooth_series(tstt_mean_hist, smooth_window)
        axes[0, 1].plot(x, tstt_mean_hist, color="#2ca02c", label="TSTT Mean")
        axes[0, 1].plot(x, tstt_mean_smooth, color="#2ca02c", linestyle="--", label="TSTT Mean (smoothed)")
        axes[0, 1].set_title("TSTT Mean")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("TSTT")
        axes[0, 1].legend()

        tstt_auc_smooth = smooth_series(tstt_auc_hist, smooth_window)
        axes[1, 0].plot(x, tstt_auc_hist, color="#9467bd", label="TSTT AUC")
        axes[1, 0].plot(x, tstt_auc_smooth, color="#9467bd", linestyle="--", label="TSTT AUC (smoothed)")
        axes[1, 0].set_title("TSTT AUC")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("AUC")
        axes[1, 0].legend()

        critic_vals = [v if v is not None else np.nan for v in critic_hist]
        critic_smooth = smooth_series(critic_vals, smooth_window)
        axes[1, 1].plot(x, critic_vals, color="#d62728", label="Critic Loss")
        axes[1, 1].plot(x, critic_smooth, color="#d62728", linestyle="--", label="Critic Loss (smoothed)")
        axes[1, 1].set_title("Critic Loss")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].legend()

        actor_vals = [v if v is not None else np.nan for v in actor_hist]
        actor_smooth = smooth_series(actor_vals, smooth_window)
        axes[2, 0].plot(x, actor_vals, color="#ff7f0e", label="Actor Loss")
        axes[2, 0].plot(x, actor_smooth, color="#ff7f0e", linestyle="--", label="Actor Loss (smoothed)")
        axes[2, 0].set_title("Actor Loss")
        axes[2, 0].set_xlabel("Episode")
        axes[2, 0].set_ylabel("Loss")
        axes[2, 0].legend()

        tstt_last_vals = [m["tstt_last"] for m in metrics]
        tstt_last_smooth = smooth_series(tstt_last_vals, smooth_window)
        axes[2, 1].plot(x, tstt_last_vals, color="#8c564b", label="TSTT Last")
        axes[2, 1].plot(x, tstt_last_smooth, color="#8c564b", linestyle="--", label="TSTT Last (smoothed)")
        axes[2, 1].set_title("TSTT Last (Episode End)")
        axes[2, 1].set_xlabel("Episode")
        axes[2, 1].set_ylabel("TSTT")
        axes[2, 1].legend()

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
