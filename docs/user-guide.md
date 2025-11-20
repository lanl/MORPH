
## User Guide for finetuning MORPH

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

### Setup the dataset 
- The dataset directory is
```
cd datasets
```
- For demo run, create a subfolder fns-kf inside "datasets" folder. Then download the data "solution_0.nc" from [PDEGym](https://huggingface.co/datasets/camlab-ethz/FNS-KF/tree/main).
- The dataset need to be in .npy/.h5/.nc formats.
- For .h5/.nc formats, convert to .npy format using
```
python src/utils/convert_nc_h5_to_npy.py --data_format nc --source_nc_file <> --target_npy_loc <>
```
- For demo run,
```
python src/utils/convert_nc_h5_to_npy.py --data_format nc --source_nc_file fns-kf/solution_0.nc --target_npy_loc fns-kf
```

### Run the finetuning code
- Finetuning scripts ask for ".npy" file.
- The scripts convert into UPTF-7 format and performs ReVIN (Reversible Instance Normalization).
- The scripts load weights from "models/FM" folder or downloads from hugging face.
- The results are saved in "experiments/results" folder

- Check all the arguments for running the script
```
python scripts/finetune_MORPH_general.py -h
```

- Demo finetune run: checkpoint exists, fns-kf data, MORPH-FM-Ti, level-1 finetuning, 5 epochs, 100 trajectories
```
python scripts/finetune_MORPH_general.py --dataset fns-kf/solution_0.npy --dataset_name fns-kf --dataset_specs 2 1 1 128 128 --model_choice FM --model_size Ti --ckpt_from FM --checkpoint morph-Ti-FM-max_ar1_ep225.pth --ft_level1 --parallel no --n_epochs 5 --n_traj 100
```

- If checkpoint is not avaiable in "models/FM", set argument --download_model

```
python scripts/finetune_MORPH_general.py --dataset fns-kf/solution_0.npy --dataset_name fns-kf --dataset_specs 2 1 1 128 128 --model_choice FM --model_size Ti --ckpt_from FM --download_model --ft_level1 --parallel no --n_epochs 5 --n_traj 100
```

- Demo inference run: set --n_epochs 0

### Results
The results of finetuning/inference will appear in 'experiments/results/test'. Types of results
1. learning curve
2. metrics (mse,rmse,mae,nrmse,vrmse) as a .txt file
3. images: next frame prediction
4. images: rollouts










































