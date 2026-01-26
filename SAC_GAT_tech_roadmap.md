# SAC+GAT 交通路网修复：可实现技术方案（面向 SCI 创新要求）

本方案面向“道路修复顺序优化”的交通路网韧性问题，以 **SAC + GAT** 为核心算法组合，使用 **BPR 数字孪生**保证物理意义完备，输出可直接落地的研发路线与接口设计。

---

## 1. 核心算法选型逻辑（数学与物理必然性）

### 1.1 SAC：熵正则的随机策略更适配非线性、非平稳修复环境
- **组合优化 + 局部最优问题**：修复序列是离散组合问题，TSTT 对动作序列高度非线性。SAC 最大化熵：
  \[
  J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim \pi} [r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))]
  \]
  通过熵项强制探索，减轻“只修附近路段”的局部最优陷阱。
- **动态性与非平稳性**：修复动作改变了网络可达性与流量分布，TSTT 随之剧烈波动。SAC 的随机策略对环境漂移更鲁棒，优于 DDPG 等确定性策略。

### 1.2 GAT：路网拓扑与关键性贡献的显式建模
- **拓扑感知**：路网是图，单段修复的全网影响取决于其结构位置。GAT 用注意力系数显式学习“关键边/节点”贡献：
  \[
  \alpha_{ij} = \text{softmax}_j\big(\text{LeakyReLU}(a^\top[W h_i \Vert W h_j])\big)
  \]
- **归纳偏置与参数效率**：相比全连接，GAT 仅对邻接信息聚合，减少参数，提升收敛速度与泛化性，符合交通图结构先验。

---

## 2. 物理建模层：BPR 数字孪生环境

### 2.1 数据源与可用输入
当前仓库已包含 `maps_simu/sioux_falls.graphml`（示例路网）。建议同时使用 TransportationNetworks 提供的 Sioux Falls `.tntp` 原始数据以获取 **OD 需求**与**路段属性**。

建议数据字段映射：
- **路段属性**：`(from_node, to_node, t0, capacity)`  
- **OD 需求矩阵**：`demand[o][d]`

### 2.2 BPR 物理约束与唯一指标
每一步修复后重新分配流量，并计算 BPR 通行时间：
\[
t_a = t_a^0\left[1 + 0.15\left(\frac{v_a}{C_a}\right)^4\right]
\]

全网总旅行时间（唯一优化目标）：
\[
TSTT = \sum_a v_a \cdot t_a
\]

---

## 3. 状态、动作与奖励（物理单调性）

### 3.1 状态空间（单调性设计）
为确保奖励上升与 TSTT 下降一致，状态特征引入对数压缩：
\[
s_t = [\log_{10}(TSTT), V/C, \text{betweenness}]
\]
- `log10(TSTT)`：提升对 1% 微小改进的梯度敏感性  
- `V/C`：反映当前拥堵强度  
- `betweenness`：捕捉拓扑关键性

### 3.2 动作空间与掩码
- 动作 = 选择一个**待修复路段**  
- **Action Masking**：已修复路段从策略分布中剔除，避免无效动作  
- 若路网规模为 `E`，策略输出长度 `E` 的 logits，并对损坏集合外位置设 `-inf`

### 3.3 边际贡献奖励（解决“奖励升但效率降”）
\[
R_t = \alpha\cdot(TSTT_{t-1}-TSTT_t)
      + \beta\cdot \mathbb{I}(\text{Complete})
      - \gamma
\]
- **边际贡献主导**：确保奖励与效率严格一致  
- **权重约束**：\(\alpha\cdot \Delta TSTT_{min} > \gamma\)，防止智能体“逃避行动”

---

## 4. 环境与数据接口设计（可实现）

### 4.1 数据读取与结构化
建议将数据解析为如下结构，统一用于 GAT 与环境：

```
GraphData:
  nodes: List[Node]
  edges: List[Edge]  # 每个 Edge 包含 (u, v, t0, capacity, is_damaged)
  od_matrix: Dict[(o, d), demand]
```

### 4.2 环境核心 API（伪代码接口）

```
class RepairEnv:
    def __init__(self, graph: GraphData, bpr_alpha=0.15, bpr_beta=4):
        ...

    def reset(self) -> State:
        self.reset_damage()
        self.compute_flow_assignment()
        return self.get_state()

    def step(self, action_edge_id) -> (State, reward, done, info):
        self.apply_repair(action_edge_id)
        self.compute_flow_assignment()
        tstt = self.compute_tstt()
        reward = self.compute_reward(tstt)
        done = self.check_done()
        return self.get_state(), reward, done, {"tstt": tstt}

    def compute_flow_assignment(self):
        # 采用 BPR + 迭代分配（如 Frank-Wolfe），输出 v_a
        ...

    def compute_tstt(self) -> float:
        return sum(v_a * t_a for each edge)
```

### 4.3 关键计算流程图（最小版）
```
损坏路网 -> action -> 修复边 -> BPR分配 -> 计算TSTT -> reward
```

---

## 5. 网络架构：全局感知 Actor（GAT + SAC）

### 5.1 编码器与全局上下文
1. **Encoder**：3 层 GAT
2. **Global Pooling**：对节点特征均值池化，形成全局向量
3. **Feature Fusion**：将全局向量与候选边特征拼接

### 5.2 Actor（带掩码）
- Actor 输入：融合后的 edge 表征  
- 输出：对每条边的 logits  
- 采样前应用 `action_mask`，无效动作置 `-inf`

### 5.3 Critic（Twin Q）
- 使用两个 Q 网络，输入 `(state, action)`  
- 共享 GAT 编码器或复制编码器均可（共享更省参）

---

## 6. 训练流程（SAC 伪代码）

```
for episode in range(N):
    s = env.reset()
    done = False
    while not done:
        a ~ pi_theta(a|s) with action_mask
        s', r, done, info = env.step(a)
        replay.add(s, a, r, s', done)
        s = s'

        if len(replay) > batch_size:
            # Critic 更新
            y = r + gamma * (min(Q1', Q2') - alpha * log_pi(a'|s'))
            update Q1, Q2
            # Actor 更新
            update pi via policy gradient (entropy regularized)
            # 温度 alpha 自适应
```

---

## 7. 实验协议与指标（满足 SCI 要求）
- **唯一优化指标**：最小化 TSTT 曲线面积  
- **稳健性**：使用 5 个随机种子做显著性检验  
- **对比基线**：贪心最短路修复、DQN、DDPG、随机策略  
- **统计指标**：平均 TSTT、收敛步数、修复效率曲线 AUC  

---

## 8. 需要补充的外部数据
仓库仅含 `sioux_falls.graphml`，若需完整实验：  
- 下载 Sioux Falls `.tntp` 原始文件（TransportationNetworks）  
- 解析 OD 需求矩阵与路段参数  
- 将解析结果转换为 `GraphData` 结构  

---

## 9. 预期实验图表与贡献点
建议输出以下图表：
- 修复步数 vs TSTT  
- 不同算法的 TSTT AUC  
- GAT 注意力热力图（展示关键路段）  

论文创新点强调：
- BPR 物理约束 + SAC 熵探索 + GAT 拓扑注意力三者闭环  
- 边际贡献奖励保证理论一致性  

