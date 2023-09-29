#!/bin/sh
#SBATCH --partition=hgx2q
#SBATCH --gres=gpu:4
#SBATCH --time=14-0:00:00
#SBATCH --mem-per-gpu=8G

#SBATCH --output=slurm_out/last_run.out

source venv/bin/activate

python3 -u train.py --config configs/train/maskdit-imagenet-10animals.yaml --num_process_per_node 4 --eval
