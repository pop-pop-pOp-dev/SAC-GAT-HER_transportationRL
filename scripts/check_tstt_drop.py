from __future__ import annotations

import argparse

import numpy as np
import yaml

from src.data.tntp_download import download_sioux_falls
from src.data.tntp_parser import load_graph_data
from src.env.repair_env import RepairEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sioux_falls.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_paths = download_sioux_falls(cfg["data_dir"])
    graph = load_graph_data(data_paths["net_path"], data_paths["trips_path"])
    env = RepairEnv(
        graph,
        damaged_ratio=cfg["damaged_ratio"],
        assignment_iters=cfg["assignment_iters"],
        assignment_method=cfg.get("assignment_method", "msa"),
        use_cugraph=cfg.get("use_cugraph", False),
        use_torch=cfg.get("use_torch_bpr", False),
        device=cfg.get("device", "cpu"),
        seed=cfg["seed"],
    )

    state = env.reset(damaged_ratio=cfg["damaged_ratio"])
    initial_tstt = env.tstt
    vc = state.edge_features[:, 2]
    mask = state.action_mask
    action = int(np.argmax(vc * mask))
    _, _, _, info = env.step(action)
    new_tstt = info["tstt"]

    print(f"Initial TSTT: {initial_tstt:.6f}")
    print(f"New TSTT: {new_tstt:.6f}")
    if abs(initial_tstt - new_tstt) < 1e-6:
        raise RuntimeError("TSTT did not change after repair. Check capacity update and assignment.")
    print("TSTT changed as expected.")


if __name__ == "__main__":
    main()
