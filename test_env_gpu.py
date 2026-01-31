
import time
import numpy as np
import torch
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

from src.env.repair_env import RepairEnv
from src.data.tntp_parser import load_graph_data

def test_gpu_env():
    print("Loading data...")
    # Adjust paths as needed
    net_path = "./data/SiouxFalls/SiouxFalls_net.tntp"
    trips_path = "./data/SiouxFalls/SiouxFalls_trips.tntp"
    graph_data = load_graph_data(net_path, trips_path)
    
    print("Initializing RepairEnv on GPU...")
    env = RepairEnv(
        graph_data=graph_data,
        use_torch=True,
        device="cuda",
        sp_backend="torch",
        force_gpu_sp=True,
        assignment_iters=60,
        assignment_method="cfw",
        debug_reward=True
    )
    
    print("Resetting environment...")
    start_time = time.time()
    obs = env.reset()
    print(f"Reset took {time.time() - start_time:.4f}s")
    
    print("Running 10 steps...")
    for i in range(10):
        # Pick a random edge to repair
        action = env.action_space.sample() if hasattr(env, "action_space") else 0
        # Find a damaged edge manually if needed
        damaged_indices = np.where(env.is_damaged > 0)[0]
        if len(damaged_indices) > 0:
            action = damaged_indices[0]
        else:
            print("No damaged edges left.")
            break
            
        print(f"Step {i+1}: Repairing edge {action}")
        step_start = time.time()
        obs, reward, done, info = env.step(action)
        step_end = time.time()
        print(f"Step {i+1} took {step_end - step_start:.4f}s. TSTT: {info.get('tstt')}")
        
    print("Test Complete.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        test_gpu_env()
    else:
        print("CUDA NOT AVAILABLE!")
