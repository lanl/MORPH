# sweep.py
import os
import sys
import subprocess

modelvariants = ["S"]
data_frac = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

# --- MORPH fine-tuning on ICF-JAG dataset ---
MORPH_FT_CMD = [
    sys.executable, "experiments/mat_prop/main_fm_wave_prop.py",
    "--hyperparameter_study",
    "--epochs", "100",
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
        env["WANDB_NAME"] = f"ft_waveprop_frac_{d}_{mv}"
        env["WANDB_RUN_GROUP"] = "morph_ft_waveprop"

        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)

# --- MORPH standalone on ICF-JAG dataset ---
MORPH_TFS_CMD = [
    sys.executable, "experiments/mat_prop/main_fm_wave_prop.py",
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
        env["WANDB_NAME"] = f"tfs_waveprop_frac_{d}_{mv}"
        env["WANDB_RUN_GROUP"] = "morph_ft_waveprop"

        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)

print("\nDone.")
