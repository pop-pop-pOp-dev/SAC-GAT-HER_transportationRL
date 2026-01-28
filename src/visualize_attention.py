from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import yaml

from src.data.tntp_download import download_sioux_falls
from src.data.tntp_parser import load_graph_data
from src.env.repair_env import RepairEnv
from src.rl.sac import DiscreteSAC
from src.train import to_torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sioux_falls.yaml")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_paths = download_sioux_falls(cfg["data_dir"])
    graph = load_graph_data(data_paths["net_path"], data_paths["trips_path"])
    env = RepairEnv(
        graph,
        damaged_ratio=cfg["damaged_ratio"],
        assignment_iters=cfg["assignment_iters"],
        use_torch=cfg.get("use_torch_bpr", False),
        device=cfg.get("device", "cpu"),
        seed=cfg["seed"],
    )

    model_path = cfg.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("model_path not found in config or file missing.")

    device = torch.device(cfg.get("device", "cpu"))
    sample_state = env.get_state()
    node_in = sample_state.node_features.shape[1]
    edge_in = sample_state.edge_features.shape[1]
    agent = DiscreteSAC(
        node_in=node_in,
        edge_in=edge_in,
        hidden=cfg["hidden_dim"],
        embed=cfg["embed_dim"],
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        target_tau=cfg["target_tau"],
    )
    agent.load(model_path, map_location=device)
    agent.actor.to(device)

    state = env.reset(damaged_ratio=cfg["damaged_ratio"])
    node_x, edge_index, edge_attr, action_mask = to_torch(state, device)
    batch = torch.zeros(node_x.size(0), dtype=torch.long, device=device)
    logits, probs, _ = agent.actor(node_x, edge_index, edge_attr, action_mask, batch, return_attention=False)
    scores = logits.detach().cpu().numpy().squeeze()
    topk = min(args.topk, len(scores))
    top_idx = np.argsort(scores)[-topk:][::-1]
    top_vals = scores[top_idx]

    plt.figure(figsize=(8, 4))
    plt.bar(range(topk), top_vals)
    plt.xlabel("Top-K edges")
    plt.ylabel("Edge Logit (Q Proxy)")
    plt.title("Edge Logits (Decision Scores)")
    out_path = os.path.join(cfg["output_dir"], "attention_topk.png")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved attention plot to {out_path}")

    graphml_path = cfg.get("graphml_path")
    if graphml_path and os.path.exists(graphml_path):
        G = nx.read_graphml(graphml_path)
        pos = {}
        for n, data in G.nodes(data=True):
            if "x" in data and "y" in data:
                pos[n] = (float(data["x"]), float(data["y"]))
        if not pos:
            pos = nx.spring_layout(G, seed=42)

        # map edge weights by node index if possible
        edge_weights = []
        for u, v in G.edges():
            try:
                u_idx = int(u) - 1
                v_idx = int(v) - 1
                e_id = env.edge_id_map.get((u_idx, v_idx))
                edge_weights.append(scores[e_id] if e_id is not None else 0.0)
            except Exception:
                edge_weights.append(0.0)

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, pos, node_size=20)
        edges = nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_weights,
            edge_cmap=plt.cm.plasma,
            width=1.5,
        )
        plt.colorbar(edges, label="Edge Logit")
        plt.axis("off")
        geo_path = os.path.join(cfg["output_dir"], "logits_geo.png")
        plt.tight_layout()
        plt.savefig(geo_path, dpi=200)
        print(f"Saved geo attention plot to {geo_path}")


if __name__ == "__main__":
    main()
