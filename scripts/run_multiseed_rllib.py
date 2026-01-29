from __future__ import annotations

import argparse
import os
import subprocess
from multiprocessing import Pool


def run_seed(args):
    seed, config, gpu_id = args
    env = os.environ.copy()
    env["SEED_OVERRIDE"] = str(seed)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd_train = ["python", "-m", "src.train_rllib", "--config", config]
    cmd_eval = ["python", "-m", "src.eval", "--config", config]
    subprocess.check_call(cmd_train, env=env)
    subprocess.check_call(cmd_eval, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sioux_falls_rllib_ppo.yaml")
    parser.add_argument("--seeds", type=str, default="43,44,45,46,47")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", type=str, default="")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    gpu_alloc = []
    if gpus:
        for i in range(len(seeds)):
            gpu_alloc.append(gpus[i % len(gpus)])
    else:
        gpu_alloc = [None] * len(seeds)

    with Pool(processes=args.workers) as pool:
        pool.map(run_seed, [(s, args.config, gpu_alloc[i]) for i, s in enumerate(seeds)])


if __name__ == "__main__":
    main()
