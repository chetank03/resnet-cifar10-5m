#!/bin/bash
# NYU HPC Cloud Bursting (course account cs_gy_6923-2026fa). Run from a terminal opened in
# https://ood.burst.hpc.nyu.edu (NYU VPN), inside the uploaded ResNet-CIFAR10 folder.
#   ./burst.sh setup            one-time: unzip data + install PyTorch into /scratch/$USER/dl-env (CPU job)
#   EPOCHS=2 ./burst.sh train   quick test on 1x A100
#   ./burst.sh train            full run
#   ./burst.sh status           queue + last lines of the newest log
set -e
ACC=cs_gy_6923-2026fa
ENV=/scratch/$USER/dl-env
cd "$(dirname "$0")/../.."  # project root (this script lives in scripts/hpc/)
case "$1" in
  setup)
    [ -d data/train_images ] || (cd data && unzip -q train_images.zip -d train_images && unzip -q test_images.zip -d test_images)
    sbatch --account=$ACC --partition=n2c48m24 --cpus-per-task=4 --mem=16GB --time=01:00:00 --output=setup_%j.out --wrap "
      set -e
      [ -x $ENV/bin/python ] || { wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh && bash /tmp/mf.sh -b -p $ENV; }
      $ENV/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
      $ENV/bin/python -c 'import torch, torchvision; print(\"OK\", torch.__version__, torchvision.__version__)'" ;;
  train)
    E=${EPOCHS:-200}
    sbatch --export=ALL,EPOCHS=$E,ENV=$ENV $([ "$E" -le 5 ] && echo --time=00:30:00) run.SBATCH ;;
  status)
    squeue -u $USER
    f=$(ls -t resnet_*.out setup_*.out 2>/dev/null | head -1)
    [ -n "$f" ] && echo "== $f" && tail -c 3000 "$f" | tr '\r' '\n' | tail -15 ;;
  *) sed -n 2,8p "$0" ;;
esac
