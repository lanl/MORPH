# sweep.py
import os
import sys
import subprocess

#modelvariants = ["S"]
modelvariants = ["Ti"]  # for new study
data_frac = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

# --- MORPH fine-tuning on ICF-JAG dataset ---
MORPH_FT_CMD = [
    sys.executable, "experiments/ft_llnl_jag/main_fm.py",
    "--hyperparameter_study",
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
        env["WANDB_NAME"] = f"ft_icf_frac_{d}_{mv}"
        env["WANDB_RUN_GROUP"] = "morph_ft_icf_Ti"

        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)

# --- MORPH standalone on ICF-JAG dataset ---
MORPH_TFS_CMD = [
    sys.executable, "experiments/ft_llnl_jag/main_fm.py",
    "--standalone",
    "--hyperparameter_study",
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
        env["WANDB_NAME"] = f"tfs_icf_frac_{d}_{mv}"
        env["WANDB_RUN_GROUP"] = "morph_ft_icf_Ti"

        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)

print("\nDone.")
