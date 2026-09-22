# sweep.py
import os
import sys
import subprocess

DATA = [0.1, 0.25, 0.5, 0.75, 1.0]
# CMD_TFS = [
#     sys.executable, "experiments/mdr/main_fm_sst.py",
#     "--hyperparameter_study",
#     "--parallel",
#     "--epochs", "50",
#     "--standalone",
#     "--train_sensor_count", "100",
#     "--test_sensor_count", "100",
# ]

# for d in DATA:
#         cmd = CMD_TFS + [
#             "--data_frac", str(d),
#         ]

#         env = os.environ.copy()
#         env["WANDB_NAME"] = f"tfs_morph_sst_data-{d}"
#         env["WANDB_RUN_GROUP"] = f"morph_sst_finetuning_2"
#         env["WANDB_PROJECT"] = f"morph_sst_finetuning_2"

#         print("\nRunning:", " ".join(cmd))
#         subprocess.run(cmd, env=env, check=True)

CMD_FT = [
    sys.executable, "experiments/mdr/main_fm_sst.py",
    "--hyperparameter_study",
    "--parallel",
    "--epochs", "50",
    "--train_sensor_count", "100",
    "--test_sensor_count", "100",
]

for d in DATA:
        cmd = CMD_FT + [
            "--data_frac", str(d),
        ]

        env = os.environ.copy()
        env["WANDB_NAME"] = f"ft_morph_sst_data-{d}"
        env["WANDB_RUN_GROUP"] = f"morph_sst_finetuning_2"
        env["WANDB_PROJECT"] = f"morph_sst_finetuning_2"

        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)

print("\nDone.")