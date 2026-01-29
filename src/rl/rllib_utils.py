from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm

from src.rl.rllib_env import RepairEnvGym


def resolve_checkpoint(model_dir: str) -> str | None:
    best_path = os.path.join(model_dir, "checkpoint_best.txt")
    last_path = os.path.join(model_dir, "checkpoint_last.txt")
    if os.path.exists(best_path):
        with open(best_path, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    if os.path.exists(last_path):
        with open(last_path, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    return None


def run_rllib_episode(algo: Algorithm, env: RepairEnvGym) -> Dict:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    tstt_curve: List[float] = []
    while not done:
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)
        total_reward += float(reward)
        tstt_curve.append(info.get("tstt", env.env.tstt))
    tstt_last = float(tstt_curve[-1]) if tstt_curve else env.env.tstt
    tstt_mean = float(np.mean(tstt_curve)) if tstt_curve else env.env.tstt
    tstt_auc = float(np.trapz(tstt_curve)) if tstt_curve else env.env.tstt
    return {
        "tstt_curve": tstt_curve,
        "reward": total_reward,
        "tstt_last": tstt_last,
        "tstt_mean": tstt_mean,
        "tstt_auc": tstt_auc,
        "auc": tstt_auc,
    }
