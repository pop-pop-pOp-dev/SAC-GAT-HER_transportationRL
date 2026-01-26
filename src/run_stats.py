from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict

import numpy as np

from src.stats import paired_ttest, summarize_results


def load_seed_results(results_dir: str) -> Dict[int, Dict]:
    seed_results = {}
    for path in glob.glob(os.path.join(results_dir, "seed_*", "eval_metrics.npy")):
        seed_str = os.path.basename(os.path.dirname(path)).replace("seed_", "")
        seed = int(seed_str)
        data = np.load(path, allow_pickle=True).item()
        seed_results[seed] = data
    return seed_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./outputs")
    parser.add_argument("--primary", type=str, default="sac")
    args = parser.parse_args()

    seed_results = load_seed_results(args.results_dir)
    summary = summarize_results(seed_results)

    # paired t-tests against primary
    tests = {}
    for method in summary["auc"].keys():
        if method == args.primary:
            continue
        a = [seed_results[s][args.primary]["auc"] for s in seed_results if args.primary in seed_results[s]]
        b = [seed_results[s][method]["auc"] for s in seed_results if method in seed_results[s]]
        if len(a) == len(b) and len(a) > 1:
            tests[method] = paired_ttest(a, b)

    output = {"summary": summary, "tests": tests}
    out_path = os.path.join(args.results_dir, "stats_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved stats summary to {out_path}")


if __name__ == "__main__":
    main()
