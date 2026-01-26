from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from src.env.repair_env import RepairEnv, EnvState


def select_random(state: EnvState) -> int:
    candidates = np.where(state.action_mask > 0)[0]
    return int(np.random.choice(candidates))


def select_max_vc(state: EnvState) -> int:
    vc = state.edge_features[:, 2]
    scores = vc * state.action_mask
    return int(np.argmax(scores))


def select_max_flow(state: EnvState) -> int:
    flow_proxy = state.edge_features[:, 2] * state.edge_features[:, 1]
    scores = flow_proxy * state.action_mask
    return int(np.argmax(scores))


def select_max_betweenness(state: EnvState, node_betweenness: np.ndarray, edge_index: np.ndarray) -> int:
    src, dst = edge_index
    edge_bw = (node_betweenness[src] + node_betweenness[dst]) / 2.0
    scores = edge_bw * state.action_mask
    return int(np.argmax(scores))


def run_episode(env: RepairEnv, policy: Callable[[EnvState], int]) -> Dict:
    tstt_curve: List[float] = []
    state = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = policy(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        tstt_curve.append(info["tstt"])
    return {"tstt_curve": tstt_curve, "reward": total_reward}


def get_baseline_policies(env: RepairEnv) -> Dict[str, Callable[[EnvState], int]]:
    node_bw = env.betweenness_vec
    edge_index = env.edge_index
    return {
        "random": select_random,
        "max_vc": select_max_vc,
        "max_flow": select_max_flow,
        "max_betweenness": lambda s: select_max_betweenness(s, node_bw, edge_index),
    }
