from __future__ import annotations

from typing import Callable, Dict, List
import copy

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


def select_greedy_one_step(env: RepairEnv, state: EnvState) -> int:
    candidates = np.where(state.action_mask > 0)[0]
    if candidates.size == 0:
        return int(np.argmax(state.action_mask))

    best_action = int(candidates[0])
    best_tstt = float("inf")

    orig_is_damaged = env.is_damaged.copy()
    orig_capacities = env.capacities.copy()
    orig_flow = env.flow.copy() if env.flow is not None else None
    orig_od_paths = copy.deepcopy(env.od_paths)
    orig_od_path_flows = copy.deepcopy(env.od_path_flows)
    orig_tstt = env.tstt
    orig_unassigned = env.unassigned_demand

    for action in candidates:
        if orig_is_damaged[action] == 0:
            continue
        env.is_damaged[action] = 0.0
        env.capacities[action] = env.initial_capacities[action]
        env.compute_flow_assignment()
        if env.tstt < best_tstt:
            best_tstt = env.tstt
            best_action = int(action)

        env.is_damaged[:] = orig_is_damaged
        env.capacities[:] = orig_capacities
        env.flow = orig_flow.copy() if orig_flow is not None else None
        env.od_paths = copy.deepcopy(orig_od_paths)
        env.od_path_flows = copy.deepcopy(orig_od_path_flows)
        env.tstt = orig_tstt
        env.unassigned_demand = orig_unassigned

    return best_action


def run_episode(
    env: RepairEnv,
    policy: Callable[[EnvState], int],
    reward_scale: float = 1.0,
    max_steps: int = 0,
) -> Dict:
    tstt_curve: List[float] = []
    state = env.reset()
    done = False
    steps = 0
    total_reward = 0.0
    while not done:
        action = policy(state)
        state, reward, done, info = env.step(action)
        total_reward += reward * reward_scale
        tstt_curve.append(info.get("tstt", env.tstt))
        steps += 1
        if max_steps > 0 and steps >= max_steps and not done:
            break
    tstt_last = float(tstt_curve[-1]) if tstt_curve else env.tstt
    tstt_mean = float(np.mean(tstt_curve)) if tstt_curve else env.tstt
    tstt_auc = float(np.trapz(tstt_curve)) if tstt_curve else env.tstt
    return {
        "tstt_curve": tstt_curve,
        "reward": total_reward,
        "tstt_last": tstt_last,
        "tstt_mean": tstt_mean,
        "tstt_auc": tstt_auc,
        "auc": tstt_auc,
    }


def get_baseline_policies(env: RepairEnv) -> Dict[str, Callable[[EnvState], int]]:
    node_bw = env.betweenness_vec
    edge_index = env.edge_index
    return {
        "random": select_random,
        "max_vc": select_max_vc,
        "max_flow": select_max_flow,
        "max_betweenness": lambda s: select_max_betweenness(s, node_bw, edge_index),
        "greedy": lambda s: select_greedy_one_step(env, s),
    }
