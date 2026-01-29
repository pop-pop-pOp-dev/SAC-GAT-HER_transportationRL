from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.env_context import EnvContext

from src.data.tntp_download import download_sioux_falls
from src.data.tntp_parser import load_graph_data
from src.env.repair_env import RepairEnv


def _load_graph_from_config(cfg: Dict[str, Any]):
    graph_data = cfg.get("graph_data")
    if graph_data is not None:
        return graph_data
    net_path = cfg.get("net_path")
    trips_path = cfg.get("trips_path")
    if not net_path or not trips_path:
        data_paths = download_sioux_falls(cfg["data_dir"])
        net_path = data_paths["net_path"]
        trips_path = data_paths["trips_path"]
    return load_graph_data(net_path, trips_path)


class RepairEnvGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvContext | Dict[str, Any]):
        cfg = dict(config)
        worker_index = getattr(config, "worker_index", 0)
        vector_index = getattr(config, "vector_index", 0)
        seed = int(cfg.get("seed", 0)) + 1000 * int(worker_index) + int(vector_index)

        graph = _load_graph_from_config(cfg)
        self.reward_scale = float(cfg.get("reward_scale", 1.0))
        self.max_steps = int(cfg.get("max_steps", 0))
        self.damaged_ratio = float(cfg.get("damaged_ratio", 0.3))
        self.env = RepairEnv(
            graph,
            damaged_ratio=self.damaged_ratio,
            assignment_iters=cfg.get("assignment_iters", 20),
            assignment_method=cfg.get("assignment_method", "msa"),
            use_cugraph=cfg.get("use_cugraph", False),
            use_torch=cfg.get("use_torch_bpr", False),
            device=cfg.get("device", "cpu"),
            sp_backend=cfg.get("sp_backend", "auto"),
            force_gpu_sp=cfg.get("force_gpu_sp", False),
            reward_mode=cfg.get("reward_mode", "log_delta"),
            reward_alpha=cfg.get("reward_alpha", 1.0),
            reward_beta=cfg.get("reward_beta", 10.0),
            reward_gamma=cfg.get("reward_gamma", 0.1),
            reward_clip=cfg.get("reward_clip", 0.0),
            capacity_damage=cfg.get("capacity_damage", 1e-3),
            unassigned_penalty=cfg.get("unassigned_penalty", 2e7),
            seed=seed,
        )

        sample_state = self.env.get_state()
        self.num_nodes = sample_state.node_features.shape[0]
        self.num_edges = sample_state.edge_features.shape[0]
        self.node_dim = sample_state.node_features.shape[1]
        self.edge_dim = sample_state.edge_features.shape[1]

        self.action_space = spaces.Discrete(self.num_edges)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Dict(
                    {
                        "node_features": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.num_nodes, self.node_dim),
                            dtype=np.float32,
                        ),
                        "edge_features": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.num_edges, self.edge_dim),
                            dtype=np.float32,
                        ),
                    }
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_edges,),
                    dtype=np.float32,
                ),
            }
        )
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        self._steps = 0
        state = self.env.reset(damaged_ratio=self.damaged_ratio)
        obs = self._state_to_obs(state)
        return obs, {"tstt": self.env.tstt}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._steps += 1
        state, reward, done, info = self.env.step(int(action))
        reward = float(reward) * self.reward_scale
        truncated = False
        if self.max_steps > 0 and self._steps >= self.max_steps and not done:
            truncated = True
        obs = self._state_to_obs(state)
        return obs, reward, bool(done), bool(truncated), info

    def _state_to_obs(self, state):
        return {
            "obs": {
                "node_features": state.node_features.astype(np.float32),
                "edge_features": state.edge_features.astype(np.float32),
            },
            "action_mask": state.action_mask.astype(np.float32),
        }

    def render(self):
        return None
