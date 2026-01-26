from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def compute_auc(curve: List[float]) -> float:
    return float(np.trapz(curve))


def paired_ttest(a: List[float], b: List[float]) -> Dict:
    t_stat, p_val = stats.ttest_rel(a, b, nan_policy="omit")
    return {"t_stat": float(t_stat), "p_value": float(p_val)}


def summarize_results(seed_results: Dict[int, Dict[str, Dict]]) -> Dict:
    methods = set()
    for _, res in seed_results.items():
        methods.update(res.keys())

    aucs = {m: [] for m in methods}
    for _, res in seed_results.items():
        for m in methods:
            if m in res:
                aucs[m].append(res[m]["auc"])

    summary = {"auc": {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in aucs.items()}}
    return summary
