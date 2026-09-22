# sweep.py
import os
import sys
import subprocess

modelvariants = ["Ti"]
data_frac = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]

# --- MORPH fine-tuning on ICF-JAG dataset ---
MORPH_FT_CMD = [
    sys.executable, "experiments/shm/main_fm_shm.py",
    "--hyperparameter_study",
    "--epochs", "50",
]

for mv in modelvariants:
    for d in data_frac:
        cmd = MORPH_FT_CMD + [
            "--model_variant", mv,
            "--data_frac", str(d),
        ]

        if mv == "L":
            cmd += ["--l1", "--l2", "--l3"]
        elif mv == "S" or mv == "Ti":
            cmd += ["--l4"]

        env = os.environ.copy()
        env["WANDB_NAME"] = f"ft_shm_frac_{d}_{mv}"
        env["WANDB_RUN_GROUP"] = "morph_ft_shm"

        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)

# --- MORPH standalone on ICF-JAG dataset ---
MORPH_TFS_CMD = [
    sys.executable, "experiments/shm/main_fm_shm.py",
    "--standalone",
    "--hyperparameter_study",
    "--epochs", "100",
]

for mv in modelvariants:
    for d in data_frac:
        cmd = MORPH_TFS_CMD + [
            "--model_variant", mv,
            "--data_frac", str(d),
        ]

        if mv == "L":
            cmd += ["--l1", "--l2", "--l3"]
        elif mv == "S" or mv == "Ti":
            cmd += ["--l4"]

        env = os.environ.copy()
        env["WANDB_NAME"] = f"tfs_shm_frac_{d}_{mv}"
        env["WANDB_RUN_GROUP"] = "morph_ft_shm"

        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)

print("\nDone.")
