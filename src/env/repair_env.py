from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from src.data.tntp_parser import GraphData, EdgeData


@dataclass
class EnvState:
    node_features: np.ndarray
    edge_features: np.ndarray
    edge_index: np.ndarray
    action_mask: np.ndarray
    log_tstt: float
    goal_mask: np.ndarray


class RepairEnv:
    def __init__(
        self,
        graph_data: GraphData,
        damaged_ratio: float = 0.3,
        bpr_alpha: float = 0.15,
        bpr_beta: float = 4.0,
        assignment_iters: int = 20,
        assignment_method: str = "msa",
        use_cugraph: bool = False,
        use_torch: bool = False,
        device: str = "cpu",
        reward_mode: str = "log_delta",
        reward_alpha: float = 1.0,
        reward_beta: float = 10.0,
        reward_gamma: float = 0.1,
        reward_clip: float = 0.0,
        capacity_damage: float = 1e-3,
        unassigned_penalty: float = 2e7,
        debug_reward: bool = False,
        debug_reward_every: int = 0,
        seed: int = 0,
    ):
        self.graph_data = graph_data
        self.bpr_alpha = bpr_alpha
        self.bpr_beta = bpr_beta
        self.assignment_iters = assignment_iters
        self.assignment_method = assignment_method.lower()
        self.use_cugraph = use_cugraph
        self.use_torch = use_torch
        self.device = device
        self.reward_mode = reward_mode
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_gamma = reward_gamma
        self.reward_clip = reward_clip
        self.capacity_damage = capacity_damage
        self.unassigned_penalty = unassigned_penalty
        self.debug_reward = debug_reward
        self.debug_reward_every = debug_reward_every
        self._debug_step = 0
        self.rng = np.random.default_rng(seed)

        self.num_nodes = graph_data.num_nodes
        self.edges = graph_data.edges
        self.num_edges = len(self.edges)
        self.edge_index = np.array([[e.u - 1 for e in self.edges], [e.v - 1 for e in self.edges]], dtype=np.int64)
        self.initial_capacities = np.array([e.capacity for e in self.edges], dtype=np.float32)
        self.capacities = self.initial_capacities.copy()
        self.t0 = np.array([e.t0 for e in self.edges], dtype=np.float32)
        self.max_capacity = float(np.max(self.initial_capacities)) if self.num_edges > 0 else 1.0
        self.max_t0 = float(np.max(self.t0)) if self.num_edges > 0 else 1.0
        self.edge_id_map = {(e.u - 1, e.v - 1): idx for idx, e in enumerate(self.edges)}
        self.initial_tstt = None
        self.total_demand = float(np.sum(list(self.graph_data.od_demand.values())))
        self.unassigned_demand = 0.0
        self.is_reset = True

        self._init_graph()
        self._init_betweenness()
        self.reset(damaged_ratio=damaged_ratio)

    def _init_graph(self):
        self.nx_graph = nx.DiGraph()
        for idx, e in enumerate(self.edges):
            self.nx_graph.add_edge(e.u - 1, e.v - 1, edge_id=idx)

    def _init_betweenness(self):
        self.betweenness = nx.betweenness_centrality(self.nx_graph, normalized=True)
        self.betweenness_vec = np.array([self.betweenness[i] for i in range(self.num_nodes)], dtype=np.float32)

    def reset(self, damaged_ratio: float = 0.3) -> EnvState:
        damaged_count = max(1, int(self.num_edges * damaged_ratio))
        self.is_damaged = np.zeros(self.num_edges, dtype=np.float32)
        damaged_indices = self.rng.choice(self.num_edges, size=damaged_count, replace=False)
        self.is_damaged[damaged_indices] = 1.0

        self.capacities = self.initial_capacities.copy()
        self.capacities[damaged_indices] = self.capacity_damage
        self.goal_mask = self.is_damaged.copy()
        self.flow = np.zeros(self.num_edges, dtype=np.float32)
        self.tstt = None
        self.compute_flow_assignment()
        self.initial_tstt = self.tstt
        self.is_reset = False
        return self.get_state()

    def step(self, action_edge_id: int) -> Tuple[EnvState, float, bool, Dict]:
        if action_edge_id < 0 or action_edge_id >= self.num_edges:
            raise ValueError(f"action_edge_id {action_edge_id} out of range (0..{self.num_edges - 1})")
        if self.is_damaged[action_edge_id] == 0:
            reward = -5.0
            return self.get_state(), reward, False, {"tstt": self.tstt}

        prev_tstt = self.tstt
        self.is_damaged[action_edge_id] = 0.0
        self.capacities[action_edge_id] = self.initial_capacities[action_edge_id]
        self.compute_flow_assignment()
        reward = self.compute_reward_with_goal(
            prev_tstt,
            self.tstt,
            self.goal_mask,
            self.is_damaged,
            alpha=self.reward_alpha,
            beta=self.reward_beta,
            gamma=self.reward_gamma,
            mode=self.reward_mode,
            clip=self.reward_clip,
        )
        if self.debug_reward:
            self._debug_step += 1
            if self.debug_reward_every <= 0 or self._debug_step % self.debug_reward_every == 0:
                diff = prev_tstt - self.tstt
                print(f"[reward_debug] prev={prev_tstt:.6g} curr={self.tstt:.6g} diff={diff:.6g} reward={reward:.6g}")
        done = self.is_goal_complete(self.goal_mask, self.is_damaged)
        return self.get_state(), reward, bool(done), {"tstt": self.tstt}

    def compute_reward(self, prev_tstt: float, curr_tstt: float, alpha: float = 1.0, beta: float = 10.0, gamma: float = 0.1) -> float:
        delta = prev_tstt - curr_tstt
        complete_bonus = beta if self.is_damaged.sum() == 0 else 0.0
        return alpha * delta + complete_bonus - gamma

    def compute_reward_with_goal(
        self,
        prev_tstt: float,
        curr_tstt: float,
        goal_mask: np.ndarray,
        damaged_mask: np.ndarray,
        alpha: float = 1.0,
        beta: float = 10.0,
        gamma: float = 0.1,
        mode: str = "delta",
        clip: float = 0.0,
    ) -> float:
        if mode == "neg_tstt":
            delta = -curr_tstt
        elif mode == "log_delta":
            delta = np.log10(max(prev_tstt, 1.0)) - np.log10(max(curr_tstt, 1.0))
        elif mode == "rel_improve":
            base = self.initial_tstt if self.initial_tstt is not None else prev_tstt
            delta = ((prev_tstt - curr_tstt) / max(base, 1.0)) * 100.0
        else:
            delta = prev_tstt - curr_tstt
        complete_bonus = beta if self.is_goal_complete(goal_mask, damaged_mask) else 0.0
        reward = alpha * delta + complete_bonus - gamma
        if clip and clip > 0:
            reward = float(np.clip(reward, -clip, clip))
        return reward

    def is_goal_complete(self, goal_mask: np.ndarray, damaged_mask: np.ndarray) -> bool:
        return bool(np.sum(goal_mask * damaged_mask) == 0.0)

    def set_goal(self, goal_mask: np.ndarray) -> None:
        self.goal_mask = goal_mask.astype(np.float32)

    def compute_flow_assignment(self):
        if self.assignment_iters <= 0:
            raise ValueError("assignment_iters must be > 0 to update TSTT.")
        if self.flow is None or self.is_reset:
            self.flow = np.zeros(self.num_edges, dtype=np.float32)
        t = self.compute_travel_time(self.flow)

        for it in range(self.assignment_iters):
            aux_flow, unassigned = self._all_or_nothing(t)
            if self.assignment_method == "fw":
                step = 2.0 / (it + 2.0)
            else:
                step = 1.0 / (it + 1.0)
            self.flow = (1 - step) * self.flow + step * aux_flow
            t = self.compute_travel_time(self.flow)
            self.unassigned_demand = unassigned

        self.tstt = self.compute_tstt(self.flow, t, self.unassigned_demand)

    def _all_or_nothing(self, t: np.ndarray) -> Tuple[np.ndarray, float]:
        aux_flow = np.zeros(self.num_edges, dtype=np.float32)
        unassigned = 0.0
        for origin in range(self.num_nodes):
            dests = [d for (o, d) in self.graph_data.od_demand.keys() if o - 1 == origin]
            if not dests:
                continue
            paths = self._shortest_paths_from_origin(origin, t)
            for dest in dests:
                demand = self.graph_data.od_demand[(origin + 1, dest)]
                path_edges = paths.get(dest - 1, [])
                if not path_edges:
                    unassigned += demand
                    continue
                for e_id in path_edges:
                    aux_flow[e_id] += demand
        return aux_flow, unassigned

    def _shortest_paths_from_origin(self, origin: int, t: np.ndarray) -> Dict[int, List[int]]:
        if self.use_cugraph:
            try:
                import cudf
                import cugraph

                df = cudf.DataFrame(
                    {
                        "src": self.edge_index[0],
                        "dst": self.edge_index[1],
                        "weight": t,
                    }
                )
                G = cugraph.DiGraph()
                G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
                sp = cugraph.shortest_path(G, source=origin)
                paths = {}
                if "predecessor" in sp.columns:
                    pred = sp.set_index("vertex")["predecessor"].to_pandas()
                    for dest, p in pred.items():
                        if dest == origin or p < 0:
                            continue
                        path_nodes = []
                        cur = dest
                        while cur != origin and cur != -1:
                            path_nodes.append(cur)
                            cur = int(pred.get(cur, -1))
                        if cur != origin:
                            continue
                        path_nodes.append(origin)
                        path_nodes = path_nodes[::-1]
                        edge_ids = []
                        for i in range(len(path_nodes) - 1):
                            edge_ids.append(self.edge_id_map[(path_nodes[i], path_nodes[i + 1])])
                        paths[dest] = edge_ids
                    return paths
            except Exception:
                pass
        # Prefer scipy sparse dijkstra for speed; fallback to networkx.
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import dijkstra

            row = self.edge_index[0]
            col = self.edge_index[1]
            weights = t.copy()
            graph = csr_matrix((weights, (row, col)), shape=(self.num_nodes, self.num_nodes))
            dist, predecessors = dijkstra(graph, directed=True, indices=origin, return_predecessors=True)
            paths = {}
            for dest in range(self.num_nodes):
                if dest == origin or predecessors[dest] < 0:
                    continue
                path_nodes = []
                cur = dest
                while cur != origin and cur != -9999:
                    path_nodes.append(cur)
                    cur = predecessors[cur]
                path_nodes.append(origin)
                path_nodes = path_nodes[::-1]
                edge_ids = []
                for i in range(len(path_nodes) - 1):
                    edge_ids.append(self.edge_id_map[(path_nodes[i], path_nodes[i + 1])])
                paths[dest] = edge_ids
            return paths
        except Exception:
            paths = {}
            for dest in range(self.num_nodes):
                if dest == origin:
                    continue
                edge_ids = self.shortest_path_edges(origin, dest, t)
                if edge_ids:
                    paths[dest] = edge_ids
            return paths

    def compute_travel_time(self, flow: np.ndarray) -> np.ndarray:
        if not self.use_torch:
            t = np.zeros(self.num_edges, dtype=np.float32)
            for i in range(self.num_edges):
                if self.is_damaged[i] == 1.0:
                    t[i] = 1e6
                    continue
                vc = max(flow[i] / max(self.capacities[i], 1e-6), 0.0)
                t[i] = self.t0[i] * (1.0 + self.bpr_alpha * (vc ** self.bpr_beta))
            return t

        import torch

        flow_t = torch.tensor(flow, dtype=torch.float32, device=self.device)
        cap_t = torch.tensor(self.capacities, dtype=torch.float32, device=self.device)
        t0_t = torch.tensor(self.t0, dtype=torch.float32, device=self.device)
        damaged_t = torch.tensor(self.is_damaged, dtype=torch.float32, device=self.device)
        vc = torch.clamp(flow_t / torch.clamp(cap_t, min=1e-6), min=0.0)
        t = t0_t * (1.0 + self.bpr_alpha * (vc ** self.bpr_beta))
        t = torch.where(damaged_t > 0, torch.full_like(t, 1e6), t)
        return t.detach().cpu().numpy()

    def compute_tstt(self, flow: np.ndarray, t: np.ndarray, unassigned_demand: float = 0.0) -> float:
        base = float(np.sum(flow * t))
        if unassigned_demand > 0:
            frac = float(unassigned_demand) / max(self.total_demand, 1.0)
            penalty = float(self.unassigned_penalty) * frac
        else:
            penalty = 0.0
        return base + penalty

    def shortest_path_edges(self, origin: int, dest: int, t: np.ndarray) -> List[int]:
        for (u, v, data) in self.nx_graph.edges(data=True):
            e_id = data["edge_id"]
            self.nx_graph[u][v]["weight"] = t[e_id]
        try:
            path = nx.shortest_path(self.nx_graph, origin, dest, weight="weight")
        except nx.NetworkXNoPath:
            return []
        edges = []
        for i in range(len(path) - 1):
            e_id = self.nx_graph[path[i]][path[i + 1]]["edge_id"]
            edges.append(e_id)
        return edges

    def get_state(self) -> EnvState:
        # dynamic betweenness based on current connectivity
        active_edges = [
            (u, v)
            for u, v, data in self.nx_graph.edges(data=True)
            if self.is_damaged[data["edge_id"]] == 0
        ]
        if active_edges:
            subgraph = self.nx_graph.edge_subgraph(active_edges)
            current_bw = nx.betweenness_centrality(subgraph, normalized=True)
            bw_vec = np.array([current_bw.get(i, 0.0) for i in range(self.num_nodes)], dtype=np.float32)
        else:
            bw_vec = np.zeros(self.num_nodes, dtype=np.float32)
        bw_max = float(np.max(bw_vec)) if bw_vec.size else 0.0
        if bw_max > 0:
            bw_vec = bw_vec / bw_max

        raw_vc = np.zeros(self.num_edges, dtype=np.float32)
        for i in range(self.num_edges):
            raw_vc[i] = self.flow[i] / max(self.capacities[i], 1e-6)
        vc = np.where(self.is_damaged > 0, 0.0, raw_vc)
        vc = np.log1p(vc)
        vc = np.clip(vc, 0.0, 5.0)

        goal_total = float(np.sum(self.goal_mask))
        remaining = float(np.sum(self.goal_mask * self.is_damaged))
        remaining_ratio = remaining / max(goal_total, 1.0)
        
        avg_flow = float(np.mean(self.flow[self.is_damaged == 0])) if np.sum(self.is_damaged == 0) > 0 else 0.0
        avg_flow_norm = avg_flow / max(self.total_demand / max(self.num_edges, 1), 1.0)
        
        node_features = np.stack(
            [
                bw_vec,
                np.full(self.num_nodes, remaining_ratio, dtype=np.float32),
                np.full(self.num_nodes, avg_flow_norm, dtype=np.float32),
            ],
            axis=1,
        )

        edge_id_norm = np.arange(self.num_edges, dtype=np.float32) / max(self.num_edges - 1, 1)

        edge_features = np.stack(
            [
                (self.t0.astype(np.float32) / max(self.max_t0, 1e-6)),
                (self.capacities.astype(np.float32) / max(self.max_capacity, 1e-6)),
                vc,
                self.is_damaged,
                self.goal_mask,
                edge_id_norm,
            ],
            axis=1,
        )

        action_mask = self.is_damaged.astype(np.float32)
        current_tstt = self.tstt if self.tstt is not None else self.initial_tstt
        log_tstt_val = float(np.log10(max(current_tstt, 1.0))) if current_tstt is not None else 0.0
        return EnvState(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=self.edge_index,
            action_mask=action_mask,
            log_tstt=log_tstt_val,
            goal_mask=self.goal_mask.copy(),
        )
