import subprocess

commands = [
    "CUDA_VISIBLE_DEVICES=0 python train_offline.py --seed=1",
    "CUDA_VISIBLE_DEVICES=1 python train_offline.py --seed=2",
    "CUDA_VISIBLE_DEVICES=2 python train_offline.py --seed=3",
    "CUDA_VISIBLE_DEVICES=3 python train_offline.py --seed=4",
    "CUDA_VISIBLE_DEVICES=4 python train_offline.py --seed=5",
    "CUDA_VISIBLE_DEVICES=5 python train_offline.py --seed=6",
    "CUDA_VISIBLE_DEVICES=6 python train_offline.py --seed=7",
    "CUDA_VISIBLE_DEVICES=7 python train_offline.py --seed=8",
]

processes = []

# Use --command instead of -e for better compatibility
for cmd in commands:
    p = subprocess.Popen(["terminator", "--command", f"bash -c '{cmd}; exec bash'"])
    processes.append(p)

# Wait for all processes to complete
for p in processes:
    p.wait()
