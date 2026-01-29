from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
import yaml
import ray
from ray.rllib.models import ModelCatalog

from src.data.tntp_download import download_sioux_falls
from src.data.tntp_parser import load_graph_data
from src.env.repair_env import RepairEnv
from src.rl.rllib_env import RepairEnvGym
from src.rl.rllib_models import GATMaskedPolicyModel, GATMaskedQModel
from src.rl.rllib_utils import run_rllib_episode


def _apply_seed_override(cfg: Dict) -> Dict:
    seed_override = os.environ.get("SEED_OVERRIDE")
    if seed_override is None:
        return cfg
    seed = int(seed_override)
    cfg = dict(cfg)
    cfg["seed"] = seed
    output_dir = cfg["output_dir"]
    cfg["output_dir"] = os.path.join(output_dir, f"seed_{seed}")
    model_dir = cfg.get("model_dir", output_dir)
    cfg["model_dir"] = os.path.join(model_dir, f"seed_{seed}")
    return cfg


def _resolve_model_config(cfg: Dict, graph) -> Dict:
    sample_env = RepairEnv(
        graph,
        damaged_ratio=cfg["damaged_ratio"],
        assignment_iters=cfg["assignment_iters"],
        assignment_method=cfg.get("assignment_method", "msa"),
        use_cugraph=cfg.get("use_cugraph", False),
        use_torch=cfg.get("use_torch_bpr", False),
        device=cfg.get("env_device", "cpu"),
        sp_backend=cfg.get("env_sp_backend", cfg.get("sp_backend", "auto")),
        force_gpu_sp=cfg.get("env_force_gpu_sp", False),
        reward_mode=cfg.get("reward_mode", "log_delta"),
        reward_alpha=cfg.get("reward_alpha", 1.0),
        reward_beta=cfg.get("reward_beta", 10.0),
        reward_gamma=cfg.get("reward_gamma", 0.1),
        reward_clip=cfg.get("reward_clip", 0.0),
        capacity_damage=cfg.get("capacity_damage", 1e-3),
        unassigned_penalty=cfg.get("unassigned_penalty", 2e7),
        seed=cfg["seed"],
    )
    state = sample_env.get_state()
    return {
        "node_in": state.node_features.shape[1],
        "edge_in": state.edge_features.shape[1],
        "hidden": cfg["hidden_dim"],
        "embed": cfg["embed_dim"],
        "num_layers": cfg.get("gat_layers", 3),
        "edge_index": state.edge_index.tolist(),
    }


def _register_models():
    ModelCatalog.register_custom_model("gat_masked_policy", GATMaskedPolicyModel)
    ModelCatalog.register_custom_model("gat_masked_q", GATMaskedQModel)


def _build_algorithm(cfg: Dict, model_cfg: Dict, env_config: Dict):
    algo = str(cfg.get("algo", "ppo")).lower()
    num_workers = int(cfg.get("num_workers", 0))
    rollout_fragment = cfg.get("rollout_fragment_length", "auto")
    num_gpus = 1 if str(cfg.get("device", "cpu")).startswith("cuda") else 0

    if algo == "ppo":
        from ray.rllib.algorithms.ppo import PPOConfig

        algo_config = (
            PPOConfig()
            .environment(env=RepairEnvGym, env_config=env_config)
            .framework("torch")
            .training(
                gamma=cfg["gamma"],
                lr=cfg["lr"],
                model={"custom_model": "gat_masked_policy", "custom_model_config": model_cfg},
                train_batch_size=cfg.get("train_batch_size", cfg.get("batch_size", 512)),
                sgd_minibatch_size=cfg.get("sgd_minibatch_size", cfg.get("batch_size", 512)),
                num_sgd_iter=cfg.get("ppo_epochs", 10),
            )
            .rollouts(num_rollout_workers=num_workers, rollout_fragment_length=rollout_fragment, batch_mode="complete_episodes")
            .resources(num_gpus=num_gpus)
        )
    elif algo == "a2c":
        try:
            from ray.rllib.algorithms.a2c import A2CConfig
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("A2CConfig not available in this RLlib version.") from exc

        algo_config = (
            A2CConfig()
            .environment(env=RepairEnvGym, env_config=env_config)
            .framework("torch")
            .training(
                gamma=cfg["gamma"],
                lr=cfg["lr"],
                model={"custom_model": "gat_masked_policy", "custom_model_config": model_cfg},
            )
            .rollouts(num_rollout_workers=num_workers, rollout_fragment_length=rollout_fragment, batch_mode="complete_episodes")
            .resources(num_gpus=num_gpus)
        )
    elif algo == "impala":
        from ray.rllib.algorithms.impala import IMPALAConfig

        algo_config = (
            IMPALAConfig()
            .environment(env=RepairEnvGym, env_config=env_config)
            .framework("torch")
            .training(
                gamma=cfg["gamma"],
                lr=cfg["lr"],
                model={"custom_model": "gat_masked_policy", "custom_model_config": model_cfg},
            )
            .rollouts(num_rollout_workers=num_workers, rollout_fragment_length=rollout_fragment, batch_mode="complete_episodes")
            .resources(num_gpus=num_gpus)
        )
    elif algo in {"dqn", "rainbow"}:
        from ray.rllib.algorithms.dqn import DQNConfig

        enable_rainbow = algo == "rainbow"
        algo_config = (
            DQNConfig()
            .environment(env=RepairEnvGym, env_config=env_config)
            .framework("torch")
            .training(
                gamma=cfg["gamma"],
                lr=cfg["lr"],
                model={"custom_model": "gat_masked_q", "custom_model_config": model_cfg},
                train_batch_size=cfg.get("train_batch_size", cfg.get("batch_size", 512)),
                replay_buffer_config={
                    "type": "MultiAgentReplayBuffer",
                    "capacity": int(cfg.get("buffer_size", 1000000)),
                },
                learning_starts=int(cfg.get("batch_start", 2000)),
                double_q=bool(cfg.get("double_q", True)),
                dueling=bool(cfg.get("dueling", True)),
                noisy=bool(cfg.get("noisy", enable_rainbow)),
                n_step=int(cfg.get("n_step", 3 if enable_rainbow else 1)),
                num_atoms=int(cfg.get("num_atoms", 51 if enable_rainbow else 1)),
                v_min=float(cfg.get("v_min", -200000.0)),
                v_max=float(cfg.get("v_max", 200000.0)),
            )
            .rollouts(num_rollout_workers=num_workers, rollout_fragment_length=rollout_fragment, batch_mode="complete_episodes")
            .resources(num_gpus=num_gpus)
        )
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    return algo_config.build()


def train(cfg: Dict):
    cfg = _apply_seed_override(cfg)
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_paths = download_sioux_falls(cfg["data_dir"])
    graph = load_graph_data(data_paths["net_path"], data_paths["trips_path"])

    env_config = {
        "data_dir": cfg["data_dir"],
        "net_path": data_paths["net_path"],
        "trips_path": data_paths["trips_path"],
        "damaged_ratio": cfg["damaged_ratio"],
        "assignment_iters": cfg["assignment_iters"],
        "assignment_method": cfg.get("assignment_method", "msa"),
        "use_cugraph": cfg.get("use_cugraph", False),
        "use_torch_bpr": cfg.get("use_torch_bpr", False),
        "device": cfg.get("env_device", "cpu"),
        "sp_backend": cfg.get("env_sp_backend", cfg.get("sp_backend", "auto")),
        "force_gpu_sp": cfg.get("env_force_gpu_sp", False),
        "reward_mode": cfg.get("reward_mode", "log_delta"),
        "reward_alpha": cfg.get("reward_alpha", 1.0),
        "reward_beta": cfg.get("reward_beta", 10.0),
        "reward_gamma": cfg.get("reward_gamma", 0.1),
        "reward_clip": cfg.get("reward_clip", 0.0),
        "capacity_damage": cfg.get("capacity_damage", 1e-3),
        "unassigned_penalty": cfg.get("unassigned_penalty", 2e7),
        "reward_scale": cfg.get("reward_scale", 1.0),
        "max_steps": cfg.get("max_steps", 0),
        "seed": seed,
    }

    model_cfg = _resolve_model_config(cfg, graph)
    _register_models()

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    algo = _build_algorithm(cfg, model_cfg, env_config)

    output_dir = cfg["output_dir"]
    model_dir = cfg.get("model_dir", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    episodes_target = int(cfg.get("episodes", 1000))
    eval_every = int(cfg.get("eval_every", 50))
    eval_seeds = cfg.get("eval_seeds") or [1001, 1002, 1003, 1004, 1005]
    save_best = bool(cfg.get("save_best", True))
    best_eval_tstt = float("inf")
    metrics: List[Dict] = []
    next_eval = eval_every

    def save_checkpoint(tag: str):
        ckpt_dir = os.path.join(model_dir, tag)
        checkpoint_path = algo.save(ckpt_dir)
        with open(os.path.join(model_dir, f"checkpoint_{tag}.txt"), "w", encoding="utf-8") as f:
            f.write(checkpoint_path)

    while True:
        result = algo.train()
        episodes_total = int(result.get("episodes_total", 0))
        timesteps_total = int(result.get("timesteps_total", 0))
        metrics.append(
            {
                "episodes_total": episodes_total,
                "timesteps_total": timesteps_total,
                "episode_reward_mean": float(result.get("episode_reward_mean", 0.0)),
            }
        )
        if eval_every > 0 and episodes_total >= next_eval:
            eval_results = []
            for seed_eval in eval_seeds:
                env_config_eval = dict(env_config)
                env_config_eval["seed"] = int(seed_eval)
                env_eval = RepairEnvGym(env_config_eval)
                eval_results.append(run_rllib_episode(algo, env_eval))
            avg_tstt = float(np.mean([r["tstt_last"] for r in eval_results]))
            avg_auc = float(np.mean([r["tstt_auc"] for r in eval_results]))
            metrics[-1]["eval_avg_tstt"] = avg_tstt
            metrics[-1]["eval_avg_auc"] = avg_auc
            if save_best and avg_tstt < best_eval_tstt:
                best_eval_tstt = avg_tstt
                save_checkpoint("best")
            next_eval += eval_every
        if episodes_total >= episodes_target:
            break

    save_checkpoint("last")
    out_path = os.path.join(output_dir, "train_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    algo.stop()
    ray.shutdown()
    print(f"[rllib] Saved metrics to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sioux_falls.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
