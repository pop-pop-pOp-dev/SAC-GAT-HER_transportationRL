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
        sp_backend: str = "auto",
        force_gpu_sp: bool = False,
        reward_mode: str = "log_delta",
        reward_alpha: float = 1.0,
        reward_beta: float = 10.0,
        reward_gamma: float = 0.1,
        reward_clip: float = 0.0,
        capacity_damage: float = 1e-3,
        unassigned_penalty: float = 2e7,
        gp_step: float = 1.0,
        gp_keep_paths: int = 3,
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
        self.sp_backend = (sp_backend or "auto").lower()
        self.force_gpu_sp = bool(force_gpu_sp)
        self.use_cupy = False
        self.use_torch_sp = False
        self._sp_backend_logged = False
        self._validate_accel()
        self.reward_mode = reward_mode
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_gamma = reward_gamma
        self.reward_clip = reward_clip
        self.capacity_damage = capacity_damage
        self.unassigned_penalty = unassigned_penalty
        self.gp_step = float(gp_step)
        self.gp_keep_paths = int(gp_keep_paths)
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

    def _validate_accel(self) -> None:
        if self.sp_backend == "cugraph":
            self.use_cugraph = True
            try:
                import cudf  # noqa: F401
                import cugraph  # noqa: F401
            except Exception as exc:
                if self.force_gpu_sp:
                    raise RuntimeError(f"cugraph unavailable: {exc}") from exc
                self.use_cugraph = False
        elif self.sp_backend == "cupy":
            self.use_cugraph = False
            try:
                import cupy  # noqa: F401
                import cupyx.scipy.sparse  # noqa: F401
                import cupyx.scipy.sparse.csgraph  # noqa: F401
                self.use_cupy = True
            except Exception as exc:
                if self.force_gpu_sp:
                    raise RuntimeError(f"cupy backend unavailable: {exc}") from exc
        elif self.sp_backend == "torch":
            self.use_cugraph = False
            try:
                import torch

                self.use_torch_sp = torch.cuda.is_available() and str(self.device).startswith("cuda")
                if self.force_gpu_sp and not self.use_torch_sp:
                    raise RuntimeError("torch GPU backend requested but CUDA is unavailable or device is CPU")
            except Exception as exc:
                if self.force_gpu_sp:
                    raise RuntimeError(f"torch backend unavailable: {exc}") from exc
        else:
            # auto: prefer cugraph, then cupy, then cpu
            if self.use_cugraph:
                try:
                    import cudf  # noqa: F401
                    import cugraph  # noqa: F401
                except Exception:
                    self.use_cugraph = False
            if not self.use_cugraph:
                try:
                    import cupy  # noqa: F401
                    import cupyx.scipy.sparse  # noqa: F401
                    import cupyx.scipy.sparse.csgraph  # noqa: F401
                    self.use_cupy = True
                except Exception as exc:
                    if self.force_gpu_sp:
                        raise RuntimeError(f"no GPU shortest-path backend available: {exc}") from exc
        if self.use_torch and str(self.device).startswith("cpu"):
            # Avoid torch overhead on CPU for BPR.
            self.use_torch = False

    def _init_betweenness(self):
        self.betweenness = nx.betweenness_centrality(self.nx_graph, normalized=True)
        self.betweenness_vec = np.array([self.betweenness[i] for i in range(self.num_nodes)], dtype=np.float32)

    def reset(self, damaged_ratio: float = 0.3) -> EnvState:
        damaged_count = max(1, int(self.num_edges * damaged_ratio))
        self.is_damaged = np.zeros(self.num_edges, dtype=np.float32)
        max_retries = 50
        damaged_indices = None
        for _ in range(max_retries):
            candidate = self.rng.choice(self.num_edges, size=damaged_count, replace=False)
            damaged_mask = np.zeros(self.num_edges, dtype=np.float32)
            damaged_mask[candidate] = 1.0
            active_edges = [
                (u, v)
                for u, v, data in self.nx_graph.edges(data=True)
                if damaged_mask[data["edge_id"]] == 0
            ]
            if not active_edges:
                continue
            subgraph = self.nx_graph.edge_subgraph(active_edges).copy()
            if nx.is_strongly_connected(subgraph):
                damaged_indices = candidate
                break
        if damaged_indices is None:
            damaged_indices = self.rng.choice(self.num_edges, size=damaged_count, replace=False)
        self.is_damaged[damaged_indices] = 1.0

        self.capacities = self.initial_capacities.copy()
        self.capacities[damaged_indices] = self.capacity_damage
        self.goal_mask = self.is_damaged.copy()
        self.flow = np.zeros(self.num_edges, dtype=np.float32)
        self.od_paths = {}
        self.od_path_flows = {}
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
        if self.assignment_method == "gp":
            self._compute_flow_assignment_gp()
            return
        t = self.compute_travel_time(self.flow)
        d_prev = None
        for it in range(self.assignment_iters):
            aux_flow, unassigned = self._all_or_nothing(t)
            d_fw = aux_flow - self.flow
            if self.assignment_method == "cfw":
                if d_prev is None:
                    direction = d_fw
                else:
                    num = float(np.dot(d_fw, d_fw - d_prev))
                    denom = float(np.dot(d_prev, d_prev)) + 1e-12
                    beta = max(0.0, num / denom)
                    direction = d_fw + beta * d_prev
                step = 2.0 / (it + 2.0)
                self.flow = np.maximum(self.flow + step * direction, 0.0)
                d_prev = direction
            else:
                if self.assignment_method == "fw":
                    step = 2.0 / (it + 2.0)
                else:
                    step = 1.0 / (it + 1.0)
                self.flow = (1 - step) * self.flow + step * aux_flow
            t = self.compute_travel_time(self.flow)
            self.unassigned_demand = unassigned

        self.tstt = self.compute_tstt(self.flow, t, self.unassigned_demand)

    def _path_cost(self, path_edges: Tuple[int, ...], t: np.ndarray) -> float:
        if not path_edges:
            return float("inf")
        return float(np.sum(t[list(path_edges)]))

    def _compute_flow_assignment_gp(self):
        t = self.compute_travel_time(self.flow)
        if self.is_reset or not self.od_paths:
            self.od_paths = {}
            self.od_path_flows = {}

        for it in range(self.assignment_iters):
            unassigned = 0.0
            step = self.gp_step if self.gp_step > 0 else (1.0 / (it + 1.0))
            for origin in range(self.num_nodes):
                dests = [d for (o, d) in self.graph_data.od_demand.keys() if o - 1 == origin]
                if not dests:
                    continue
                paths_dict = self._shortest_paths_from_origin(origin, t)
                for dest in dests:
                    demand = self.graph_data.od_demand[(origin + 1, dest)]
                    sp_edges = paths_dict.get(dest - 1, [])
                    if not sp_edges:
                        unassigned += demand
                        continue
                    key = (origin + 1, dest)
                    sp_tuple = tuple(sp_edges)
                    if key not in self.od_paths:
                        self.od_paths[key] = [sp_tuple]
                        self.od_path_flows[key] = [float(demand)]
                        continue
                    if sp_tuple not in self.od_paths[key]:
                        self.od_paths[key].append(sp_tuple)
                        self.od_path_flows[key].append(0.0)
                    costs = [self._path_cost(p, t) for p in self.od_paths[key]]
                    min_idx = int(np.argmin(costs))
                    flows = self.od_path_flows[key]
                    if len(flows) > 1:
                        moved = 0.0
                        for i in range(len(flows)):
                            if i == min_idx:
                                continue
                            transfer = step * flows[i]
                            flows[i] -= transfer
                            moved += transfer
                        flows[min_idx] += moved
                    if self.gp_keep_paths > 0 and len(self.od_paths[key]) > self.gp_keep_paths:
                        keep = np.argsort(costs)[: self.gp_keep_paths]
                        new_paths = [self.od_paths[key][i] for i in keep]
                        new_flows = [flows[i] for i in keep]
                        total = float(np.sum(new_flows))
                        if total > 0:
                            new_flows = [f * demand / total for f in new_flows]
                        else:
                            new_flows = [0.0 for _ in new_flows]
                            new_flows[0] = float(demand)
                        self.od_paths[key] = new_paths
                        self.od_path_flows[key] = new_flows

            flow = np.zeros(self.num_edges, dtype=np.float32)
            for key, paths in self.od_paths.items():
                flows = self.od_path_flows[key]
                for p, f in zip(paths, flows):
                    if f <= 0:
                        continue
                    for e_id in p:
                        flow[e_id] += f

            self.flow = flow
            self.unassigned_demand = unassigned
            t = self.compute_travel_time(self.flow)

        self.tstt = self.compute_tstt(self.flow, t, self.unassigned_demand)

    def _all_or_nothing(self, t: np.ndarray) -> Tuple[np.ndarray, float]:
        aux_flow = np.zeros(self.num_edges, dtype=np.float32)
        unassigned = 0.0
        if self.use_cugraph:
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

        if self.use_cupy:
            try:
                import cupy as cp
                from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
                from cupyx.scipy.sparse.csgraph import dijkstra as cp_dijkstra

                row = cp.asarray(self.edge_index[0])
                col = cp.asarray(self.edge_index[1])
                weights = cp.asarray(t, dtype=cp.float32)
                graph = cp_csr_matrix((weights, (row, col)), shape=(self.num_nodes, self.num_nodes))
                _, predecessors = cp_dijkstra(
                    graph,
                    directed=True,
                    indices=cp.arange(self.num_nodes),
                    return_predecessors=True,
                )
                pred_host = cp.asnumpy(predecessors)
                for origin in range(self.num_nodes):
                    dests = [d for (o, d) in self.graph_data.od_demand.keys() if o - 1 == origin]
                    if not dests:
                        continue
                    pred_row = pred_host[origin]
                    for dest in dests:
                        demand = self.graph_data.od_demand[(origin + 1, dest)]
                        path_edges = self._path_edges_from_predecessors(origin, dest - 1, pred_row)
                        if not path_edges:
                            unassigned += demand
                            continue
                        for e_id in path_edges:
                            aux_flow[e_id] += demand
                if not self._sp_backend_logged:
                    print("\033[92m[env] shortest-path backend: CuPy (GPU)\033[0m")
                    self._sp_backend_logged = True
                return aux_flow, unassigned
            except Exception as exc:
                if self.force_gpu_sp:
                    raise RuntimeError(f"cupy shortest-path failed: {exc}") from exc

        if self.use_torch_sp:
            return self._all_or_nothing_torch(t)

        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import dijkstra

            row = self.edge_index[0]
            col = self.edge_index[1]
            weights = t.copy()
            graph = csr_matrix((weights, (row, col)), shape=(self.num_nodes, self.num_nodes))
            _, predecessors = dijkstra(graph, directed=True, indices=range(self.num_nodes), return_predecessors=True)
            for origin in range(self.num_nodes):
                dests = [d for (o, d) in self.graph_data.od_demand.keys() if o - 1 == origin]
                if not dests:
                    continue
                pred_row = predecessors[origin]
                for dest in dests:
                    demand = self.graph_data.od_demand[(origin + 1, dest)]
                    path_edges = self._path_edges_from_predecessors(origin, dest - 1, pred_row)
                    if not path_edges:
                        unassigned += demand
                        continue
                    for e_id in path_edges:
                        aux_flow[e_id] += demand
            return aux_flow, unassigned
        except Exception:
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

    def _all_or_nothing_torch(self, t: np.ndarray) -> Tuple[np.ndarray, float]:
        import torch

        device = torch.device(self.device)
        n = self.num_nodes
        inf = torch.tensor(1e12, device=device, dtype=torch.float32)
        dist = torch.full((n, n), inf, device=device, dtype=torch.float32)
        next_hop = torch.full((n, n), -1, device=device, dtype=torch.int64)
        idx = torch.arange(n, device=device)
        dist[idx, idx] = 0.0

        row = torch.tensor(self.edge_index[0], device=device, dtype=torch.long)
        col = torch.tensor(self.edge_index[1], device=device, dtype=torch.long)
        weight = torch.tensor(t, device=device, dtype=torch.float32)
        dist[row, col] = weight
        next_hop[row, col] = col

        for k in range(n):
            alt = dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0)
            mask = alt < dist
            dist = torch.where(mask, alt, dist)
            nk = next_hop[:, k].unsqueeze(1).expand_as(next_hop)
            next_hop = torch.where(mask, nk, next_hop)

        next_cpu = next_hop.detach().cpu().numpy()
        aux_flow = np.zeros(self.num_edges, dtype=np.float32)
        unassigned = 0.0

        for (o, d), demand in self.graph_data.od_demand.items():
            origin = o - 1
            dest = d - 1
            if origin == dest:
                continue
            path_edges: List[int] = []
            cur = origin
            hops = 0
            while cur != dest and cur != -1 and hops < n:
                nxt = int(next_cpu[cur, dest])
                if nxt < 0:
                    path_edges = []
                    break
                path_edges.append(self.edge_id_map[(cur, nxt)])
                cur = nxt
                hops += 1
            if cur != dest:
                unassigned += demand
                continue
            for e_id in path_edges:
                aux_flow[e_id] += demand

        if not self._sp_backend_logged:
            print("\033[92m[env] shortest-path backend: Torch (GPU)\033[0m")
            self._sp_backend_logged = True
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
                if hasattr(cugraph, "DiGraph"):
                    G = cugraph.DiGraph()
                else:
                    try:
                        G = cugraph.Graph(directed=True)
                    except TypeError:
                        G = cugraph.Graph()
                G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
                sp = cugraph.sssp(G, source=origin)
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
                    if not self._sp_backend_logged:
                        print("\033[92m[env] shortest-path backend: cuGraph (GPU)\033[0m")
                        self._sp_backend_logged = True
                    return paths
            except Exception as exc:
                if not self._sp_backend_logged:
                    print(f"\033[91m[env] cuGraph failed, falling back to CPU: {exc}\033[0m")
                    self._sp_backend_logged = True
                self.use_cugraph = False
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
            if not self._sp_backend_logged:
                print("\033[93m[env] shortest-path backend: SciPy (CPU)\033[0m")
                self._sp_backend_logged = True
            return paths
        except Exception:
            paths = {}
            for dest in range(self.num_nodes):
                if dest == origin:
                    continue
                edge_ids = self.shortest_path_edges(origin, dest, t)
                if edge_ids:
                    paths[dest] = edge_ids
            if not self._sp_backend_logged:
                print("\033[93m[env] shortest-path backend: NetworkX (CPU)\033[0m")
                self._sp_backend_logged = True
            return paths

    def compute_travel_time(self, flow: np.ndarray) -> np.ndarray:
        if not self.use_torch:
            flow_np = np.asarray(flow, dtype=np.float32)
            cap = np.maximum(self.capacities, 1e-6)
            vc = np.clip(flow_np / cap, 0.0, 4.0)
            t = self.t0 * (1.0 + self.bpr_alpha * (vc ** self.bpr_beta))
            damaged_mask = self.is_damaged > 0.5
            t = t.astype(np.float32)
            t[damaged_mask] = 1e6
            return t

        import torch

        flow_t = torch.tensor(flow, dtype=torch.float32, device=self.device)
        cap_t = torch.tensor(self.capacities, dtype=torch.float32, device=self.device)
        t0_t = torch.tensor(self.t0, dtype=torch.float32, device=self.device)
        damaged_t = torch.tensor(self.is_damaged, dtype=torch.float32, device=self.device)
        vc = torch.clamp(flow_t / torch.clamp(cap_t, min=1e-6), min=0.0, max=4.0)
        t = t0_t * (1.0 + self.bpr_alpha * (vc ** self.bpr_beta))
        t = torch.where(damaged_t > 0, torch.full_like(t, 1e6), t)
        return t.detach().cpu().numpy()

    def _path_edges_from_predecessors(self, origin: int, dest: int, pred_row: np.ndarray) -> List[int]:
        if dest == origin or pred_row[dest] < 0:
            return []
        path_nodes = []
        cur = dest
        while cur != origin and cur != -9999:
            path_nodes.append(cur)
            cur = int(pred_row[cur])
        if cur != origin:
            return []
        path_nodes.append(origin)
        path_nodes = path_nodes[::-1]
        edge_ids = []
        for i in range(len(path_nodes) - 1):
            edge_ids.append(self.edge_id_map[(path_nodes[i], path_nodes[i + 1])])
        return edge_ids

    def compute_tstt(self, flow: np.ndarray, t: np.ndarray, unassigned_demand: float = 0.0) -> float:
        flow_np = np.asarray(flow, dtype=np.float32)
        t_np = np.asarray(t, dtype=np.float32)
        base = float(np.sum(flow_np * t_np))
        total_demand = max(self.total_demand, 1.0)
        att_base = base / total_demand
        if unassigned_demand > 0:
            frac = float(unassigned_demand) / total_demand
            penalty = float(self.unassigned_penalty) * frac
        else:
            penalty = 0.0
        return att_base + penalty

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
        vc = np.clip(vc, 0.0, 10.0)

        goal_total = float(np.sum(self.goal_mask))
        remaining = float(np.sum(self.goal_mask * self.is_damaged))
        remaining_ratio = remaining / max(goal_total, 1.0)
        
        avg_flow = float(np.mean(self.flow[self.is_damaged == 0])) if np.sum(self.is_damaged == 0) > 0 else 0.0
        avg_flow_norm = avg_flow / max(self.total_demand / max(self.num_edges, 1), 1.0)
        
        current_tstt = self.tstt if self.tstt is not None else self.initial_tstt
        log_tstt_val = float(np.log10(max(current_tstt, 1.0))) if current_tstt is not None else 0.0

        node_features = np.stack(
            [
                bw_vec,
                np.full(self.num_nodes, remaining_ratio, dtype=np.float32),
                np.full(self.num_nodes, avg_flow_norm, dtype=np.float32),
                np.full(self.num_nodes, log_tstt_val, dtype=np.float32),
            ],
            axis=1,
        )

        edge_id_norm = np.arange(self.num_edges, dtype=np.float32) / max(self.num_edges - 1, 1)

        t0_norm = np.log10(self.t0 + 1.0) / np.log10(self.max_t0 + 1.0)
        cap_norm = np.log10(self.capacities + 1.0) / np.log10(self.max_capacity + 1.0)
        edge_features = np.stack(
            [
                t0_norm.astype(np.float32),
                cap_norm.astype(np.float32),
                vc,
                self.is_damaged,
                self.goal_mask,
                edge_id_norm,
            ],
            axis=1,
        )

        action_mask = self.is_damaged.astype(np.float32)
        return EnvState(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=self.edge_index,
            action_mask=action_mask,
            log_tstt=log_tstt_val,
            goal_mask=self.goal_mask.copy(),
        )
