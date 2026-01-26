# SAC-GAT-HER_transportationRL

SAC + GAT Traffic Network Repair

完整实现：TNTP Sioux Falls 下载解析、BPR 数字孪生环境、SAC+GAT 训练与评估。

## 安装
```
pip install -r requirements.txt
```

## 训练
```
python -m src.train --config configs/sioux_falls.yaml
```

## 实时可视化（TensorBoard）
```
tensorboard --logdir outputs --bind_all
```

## 评估（示例基线）
```
python -m src.eval --config configs/sioux_falls.yaml
```

## 多基线 + 多种子并行
```
python scripts/run_multiseed.py --config configs/sioux_falls.yaml --seeds 0,1,2,3,4 --workers 2
python -m src.run_stats --results_dir outputs --primary sac
```

## 注意力可视化
```
python -m src.visualize_attention --config configs/sioux_falls.yaml --topk 20
```

## 结构说明
- `src/data/`：TNTP 下载与解析
- `src/env/`：BPR 修复环境与 TSTT
- `src/models/`：GAT 编码器
- `src/rl/`：离散 SAC
- `src/baselines.py`：基线策略
- `src/run_stats.py`：AUC 与统计检验
- `src/visualize_attention.py`：注意力可视化
- `configs/`：超参配置
- `outputs/`：训练/评估输出

## 备注
- 默认使用 Sioux Falls TNTP 数据（自动下载）
- BPR 分配支持 `msa` / `fw`（config: `assignment_method`）
- Replay Buffer 启用 PER（config: `per_alpha/per_beta`）
- 可选 geo 热力图（config: `graphml_path`）
