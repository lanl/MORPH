----------
##### Note: EIDR number O#4999 - MORPH: Shape-agnostic PDE Foundational Models. This program is Open-Source under the BSD-3 License.
----------
## MORPH: PDE foundational models
<p align="center">
  <img src="fm_vit.png" width="500" alt="Architecture of the FM">
</p>

If you use MORPH in your research, please cite:
```
@misc{rautela2025morphshapeagnosticpdefoundation,
  title={{MORPH}: Shape-agnostic {PDE} Foundation Models},
  author={Mahindra Singh Rautela and Alexander Most and Siddharth Mansingh and Bradley C. Love and Ayan Biswas and Diane Oyen and Earl Lawrence},
  year={2025},
  eprint={2509.21670},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2509.21670}
}
```
### Clone the repository
To clone the repository, click on top-right 'code' and select 'clone with HTTPS' and copy the code path and paste in the terminal.
```
git clone https://github.com/lanl/MORPH.git
```
Go to the directory
```
cd MORPH
```

### Install the requirements
- Install dependencies via environment.yml
```
conda env create -f environment.yml
```
- Activate the environment
```
conda activate pytorch_py38_env
```
- Install pytorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118                    
```
- Check pytorch installation
```
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```
Output: 
PyTorch version: 2.4.1+cu118
CUDA available: True

### Run the scripts
#### 1. Pretraining script

- Check arguments:
```
python scripts/pretrain_MORPH.py -h 
```
Output:
[--dataset {MHD,DR,CFD1D,CFD2D-IC,CFD3D,SW,DR1D,CFD2D,CFD3D-TURB,BE1D,GSDR2D,TGC3D,FNS_KF_2D,FM}]
[--chunk_mhd CHUNK_MHD] [--chunk_dr CHUNK_DR] [--chunk_cfd1d CHUNK_CFD1D]
[--chunk_cfd2dic CHUNK_CFD2DIC] [--chunk_cfd3d CHUNK_CFD3D] [--chunk_sw CHUNK_SW]
[--chunk_dr1d CHUNK_DR1D] [--chunk_cfd2d CHUNK_CFD2D] [--chunk_cfd3d_turb CHUNK_CFD3D_TURB]
[--chunk_be1d CHUNK_BE1D] [--chunk_gsdr2d CHUNK_GSDR2D] [--chunk_tgc3d CHUNK_TGC3D]
[--chunk_fnskf2d CHUNK_FNSKF2D] [--model_size {Ti,S,M,Lt,L,XL}] [--max_ar_order MAX_AR_ORDER]
[--activated_ar1k] [--ar_order AR_ORDER] [--resume] [--ckpt_name CKPT_NAME] [--finetune_ar1k]
[--tf_params filters dim heads depth mlp_dim] [--tf_reg dropout emb_dropout] [--heads_xa HEADS_XA]
[--learning_rate LEARNING_RATE] [--new_lr_ckpt NEW_LR_CKPT] [--num_epochs NUM_EPOCHS]
[--warm_epochs WARM_EPOCHS] [--patience PATIENCE]
[--bs MHD DR CFD1D CFD2D-IC CFD3D SW DR1D CFD2D CFD3D-TURB BE1D GSDR2D TGC3D FNS_KF_2D]
[--parallel {ddp,dp,no}] [--scale_gpu_utils {1x,2x,4x,0.5x,0.25x}]
[--cpu_cores_per_node CPU_CORES_PER_NODE] [--local_rank LOCAL_RANK] [--device_idx DEVICE_IDX]
[--num_workers NUM_WORKERS] [--pin_flag] [--persist_flag] [--save_every SAVE_EVERY]
[--save_batch_ckpt] [--save_batch_freq SAVE_BATCH_FREQ] [--overwrite_weights]

#### 2. Finetuning script

Check arguments:
```
python scripts/finetune_MORPH.py -h
```
Output:
[--dataset_root DATASET_ROOT] [--model_choice MODEL_CHOICE] [--model_size {Ti,S,M,L}] [--ckpt_from {FM,FT}] [--checkpoint CHECKPOINT]
[--ft_dataset {DR1D,CFD2D,CFD3D-TURB,BE1D,GSDR2D,TGC3D}] [--ft_level1] [--ft_level2]
[--ft_level3] [--ft_level4] [--lr_level4 LR_LEVEL4] [--wd_level4 WD_LEVEL4]
[--parallel {dp,no}] [--rank_lora_attn RANK_LORA_ATTN] [--rank_lora_mlp RANK_LORA_MLP]
[--lora_p LORA_P] [--n_epochs N_EPOCHS] [--n_traj N_TRAJ] [--rollout_horizon ROLLOUT_HORIZON]
[--batch_size BATCH_SIZE] [--tf_reg dropout emb_dropout] [--heads_xa HEADS_XA]
[--ar_order AR_ORDER] [--max_ar_order MAX_AR_ORDER] [--test_sample TEST_SAMPLE]
[--device_idx DEVICE_IDX] [--patience PATIENCE] [--overwrite_weights]
[--save_every SAVE_EVERY] [--save_batch_ckpt] [--save_batch_freq SAVE_BATCH_FREQ]

#### 3. Inference script

Check arguments:
```
python scripts/infer_MORPH.py -h
```
usage:
[--model_choice {MHD,DR,CFD1D,CFD2D-IC,CFD3D,SW,DR1D,CFD2D,CFD3D-TURB,BE1D,GSDR2D,TGC3D,FM}]
[--model_size {Ti,S,M,L}] [--checkpoint CHECKPOINT]
[--test_dataset {MHD,DR,CFD1D,CFD2D-IC,CFD3D,SW,DR1D,CFD2D,CFD3D-TURB,BE1D,GSDR2D,TGC3D}]
[--ar_order AR_ORDER] [--rollout_horizon ROLLOUT_HORIZON] [--device_idx DEVICE_IDX]
[--batch_size BATCH_SIZE] [--test_sample TEST_SAMPLE] [--tf_reg dropout emb_dropout]
[--heads_xa HEADS_XA] [--max_ar_order MAX_AR_ORDER]








