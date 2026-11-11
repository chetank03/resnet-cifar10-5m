#!/bin/bash
# ResNet-CIFAR10 on NYU HPC Cloud Bursting. Submit with ./burst.sh train (see burst.sh).
#SBATCH --job-name=resnet-cifar
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=resnet_%j.out

ENV=${ENV:-/scratch/$USER/dl-env}
cd "$SLURM_SUBMIT_DIR"
nvidia-smi
"$ENV/bin/python" -u train.py
