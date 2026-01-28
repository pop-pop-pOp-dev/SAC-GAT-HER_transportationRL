from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import yaml

from src.baselines import get_baseline_policies, run_episode
from src.data.tntp_download import download_sioux_falls
from src.data.tntp_parser import load_graph_data
from src.env.repair_env import RepairEnv


def save_results(all_results: Dict[str, Dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, "eval_metrics.npy")
    json_path = os.path.join(output_dir, "eval_metrics.json")
    np.save(npy_path, all_results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[eval] Saved metrics to {npy_path} and {json_path}")


def evaluate(cfg):
    seed_override = os.environ.get("SEED_OVERRIDE")
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
        cfg["output_dir"] = os.path.join(cfg["output_dir"], f"seed_{cfg['seed']}")
        cfg["model_path"] = os.path.join(cfg["output_dir"], "model.pt")
        eval_seeds = [cfg["seed"]]
    else:
        eval_seeds = cfg.get("eval_seeds")
        if eval_seeds is None:
            eval_seeds = [cfg["seed"] + i for i in range(1, 11)]

    data_paths = download_sioux_falls(cfg["data_dir"])
    graph = load_graph_data(data_paths["net_path"], data_paths["trips_path"])
    model_path = cfg.get("model_path")

    reward_scale = float(cfg.get("reward_scale", 1.0))
    max_steps = int(cfg.get("max_steps", 0)) if cfg.get("max_steps", 0) is not None else 0
    all_results: Dict[str, Dict] = {}
    for seed in eval_seeds:
        print(f"[eval] Start seed {seed}")
        env = RepairEnv(
            graph,
            damaged_ratio=cfg["damaged_ratio"],
            assignment_iters=cfg["assignment_iters"],
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

        results: Dict[str, Dict] = {}
        baselines = get_baseline_policies(env)
        for name, policy in baselines.items():
            result = run_episode(env, policy, reward_scale=reward_scale, max_steps=max_steps)
            results[name] = result
            all_results[f"seed_{seed}"] = results
            save_results(all_results, cfg["output_dir"])
            print(f"[eval] Seed {seed} baseline {name} done")

        if model_path and os.path.exists(model_path):
            from src.rl.sac import DiscreteSAC
            from src.train import to_torch

            device = torch.device(cfg.get("device", "cpu"))
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
                gamma=cfg["gamma"],
                target_tau=cfg["target_tau"],
            )
            agent.load(model_path, map_location=device)
            agent.actor.to(device)
            env.reset(damaged_ratio=cfg["damaged_ratio"])
            tstt_curve: List[float] = []
            total_reward = 0.0
            state = env.get_state()
            done = False
            steps = 0
            while not done:
                node_x, edge_index, edge_attr, action_mask = to_torch(state, device)
                out = agent.select_action(node_x, edge_index, edge_attr, action_mask, deterministic=True)
                state, reward, done, info = env.step(out.action)
                total_reward += reward * reward_scale
                tstt_curve.append(info.get("tstt", env.tstt))
                steps += 1
                if max_steps > 0 and steps >= max_steps and not done:
                    break
            tstt_last = float(tstt_curve[-1]) if tstt_curve else env.tstt
            tstt_mean = float(np.mean(tstt_curve)) if tstt_curve else env.tstt
            tstt_auc = float(np.trapz(tstt_curve)) if tstt_curve else env.tstt
            results["sac"] = {
                "tstt_curve": tstt_curve,
                "reward": total_reward,
                "tstt_last": tstt_last,
                "tstt_mean": tstt_mean,
                "tstt_auc": tstt_auc,
                "auc": tstt_auc,
            }
            all_results[f"seed_{seed}"] = results
            save_results(all_results, cfg["output_dir"])
            print(f"[eval] Seed {seed} baseline sac done")

        all_results[f"seed_{seed}"] = results
        save_results(all_results, cfg["output_dir"])
        print(f"[eval] Seed {seed} done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sioux_falls.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg)


if __name__ == "__main__":
    main()
