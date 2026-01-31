import argparse
import os
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import multiprocessing as mp
import time
from tqdm import tqdm
from src.data.tntp_download import download_sioux_falls
from src.data.tntp_parser import load_graph_data
from src.env.repair_env import RepairEnv
from src.baselines import get_baseline_policies, run_episode
from src.rl.sac import DiscreteSAC

def load_sac_agent(cfg, device):
    """Load the trained SAC agent from the checkpoint."""
    data_paths = download_sioux_falls(cfg["data_dir"])
    graph = load_graph_data(data_paths["net_path"], data_paths["trips_path"])
    
    # Create a dummy env to get state dimensions
    dummy_env = RepairEnv(graph, damaged_ratio=cfg["damaged_ratio"])
    state = dummy_env.reset()
    node_in = state.node_features.shape[1]
    edge_in = state.edge_features.shape[1]

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
    
    model_dir = cfg.get("model_dir", cfg["output_dir"])
    model_path = os.path.join(model_dir, "model_last.pt")
    if os.path.exists(model_path):
        print(f"Loading SAC model from {model_path}")
        agent.load(model_path, map_location=device)
        agent.actor.to(device)
        agent.actor.eval()
        return agent
    else:
        print(f"Warning: SAC model not found at {model_path}. Skipping SAC evaluation.")
        return None

def worker_eval(name, policy_fn_name, cfg, output_dir, device_str):
    """
    Worker function to run a single policy evaluation.
    We need to reconstruct the env inside the process to avoid pickling issues with complex objects.
    """
    # Re-seed for reproducibility per worker
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Device handling for worker
    device = torch.device(device_str)
    
    # Setup Environment
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
        gp_step=cfg.get("gp_step", 1.0),
        gp_keep_paths=cfg.get("gp_keep_paths", 3),
        reward_mode=cfg.get("reward_mode", "delta"),
        reward_alpha=cfg.get("reward_alpha", 1.0),
        reward_beta=cfg.get("reward_beta", 10.0),
        reward_gamma=cfg.get("reward_gamma", 0.1),
        reward_clip=cfg.get("reward_clip", 0.0),
        capacity_damage=cfg.get("capacity_damage", 1e-3),
        unassigned_penalty=cfg.get("unassigned_penalty", 2e7),
        fixed_damage=True,
        fixed_damage_seed=42,  # Enforce same seed
        seed=seed,
    )
    
    # Re-get policies to find the target one
    # Note: For SAC, we need to handle it carefully since it's a model
    policies = get_baseline_policies(env)
    
    policy = None
    if name == "SAC":
        sac_agent = load_sac_agent(cfg, device)
        if sac_agent:
            def sac_policy(state):
                node_x = torch.tensor(state.node_features, dtype=torch.float32, device=device)
                edge_index = torch.tensor(state.edge_index, dtype=torch.long, device=device)
                edge_attr = torch.tensor(state.edge_features, dtype=torch.float32, device=device)
                action_mask = torch.tensor(state.action_mask, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out = sac_agent.select_action(node_x, edge_index, edge_attr, action_mask, deterministic=True)
                return out.action
            policy = sac_policy
    else:
        policy = policies.get(name)

    if policy is None:
        print(f"[{name}] Policy not found or failed to load. Skipping.")
        return None

    print(f"[{name}] Starting evaluation...")
    env.reset(damaged_ratio=cfg["damaged_ratio"])
    res = run_episode(env, policy, reward_scale=cfg.get("reward_scale", 1.0))
    print(f"[{name}] Finished: TSTT Last={res['tstt_last']:.2f}, AUC={res['tstt_auc']:.2f}")
    
    # Save individual result immediately (Async write)
    res_path = os.path.join(output_dir, f"result_{name}.yaml")
    
    # Convert numpy types to python native types for safe YAML dumping
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError
        
    # Simplified result for YAML (excluding full curve to keep it readable, or include if needed)
    # User asked for "results", usually implies metrics. We can save full curve too.
    save_data = {
        "name": name,
        "tstt_last": float(res["tstt_last"]),
        "tstt_auc": float(res["tstt_auc"]),
        "tstt_mean": float(res["tstt_mean"]),
        "reward": float(res["reward"]),
        "tstt_curve": [float(x) for x in res["tstt_curve"]]
    }
    
    with open(res_path, "w") as f:
        yaml.dump(save_data, f)
        
    return name, res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sioux_falls.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Force baseline settings to match main experiment
    seed = cfg.get("seed", 42)
    cfg["seed"] = seed
    cfg["fixed_damage"] = True
    cfg["fixed_damage_seed"] = 42
    
    output_dir = "baselines_results"
    os.makedirs(output_dir, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running baselines on {device_str}")
    
    # Identify policies to run
    # We do a lightweight env init just to get keys
    dummy_paths = download_sioux_falls(cfg["data_dir"])
    dummy_graph = load_graph_data(dummy_paths["net_path"], dummy_paths["trips_path"])
    dummy_env = RepairEnv(dummy_graph)
    base_policies = list(get_baseline_policies(dummy_env).keys())
    policy_names = base_policies + ["SAC"]
    
    # Prepare arguments for workers
    ctx = mp.get_context("spawn")
    processes = []
    
    print(f"Launching {len(policy_names)} processes for: {policy_names}")
    
    for name in policy_names:
        p = ctx.Process(target=worker_eval, args=(name, name, cfg, output_dir, device_str))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    print(f"\nAll evaluations complete. Individual results saved to {output_dir}")

    # Optional: Aggregation step (can be done here or separately)
    # We can try to read back files and make the plot
    results = {}
    for name in policy_names:
        res_path = os.path.join(output_dir, f"result_{name}.yaml")
        if os.path.exists(res_path):
            with open(res_path, "r") as f:
                data = yaml.safe_load(f)
                results[name] = data

    if results:
        plt.figure(figsize=(10, 6))
        for name, res in results.items():
            curve = res["tstt_curve"]
            plt.plot(curve, label=f"{name} (AUC: {res['tstt_auc']:.0f})", linewidth=2 if name=="SAC" else 1.5)
        
        plt.title("Repair Strategy Comparison: TSTT Reduction Over Steps")
        plt.xlabel("Repair Steps")
        plt.ylabel("Total System Travel Time (TSTT)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "baseline_comparison.png"), dpi=300)
        plt.close()
        print(f"Combined plot saved to {os.path.join(output_dir, 'baseline_comparison.png')}")

if __name__ == "__main__":
    main()
