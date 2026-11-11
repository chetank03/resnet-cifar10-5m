#!/bin/bash
# Run 19 (HPC): the 0.902 recipe (run 10 / run_kd4.ps1) with Cutout-16 replacing the small random erase (Cutout is reported to help more on shifted data).
# Same 4 teachers, pure KL at T=1, 400 epochs, CutMix, then 10-view TTA. Output: sub_kdcut_tta.csv.  Resumable (RESUME=1).
# Usage: nohup bash scripts/hpc/hpc_kdcut.sh > /dev/null 2>&1 &   (from ~/ResNet-CIFAR10)   Progress: grep -a 'val acc' kdcut.log | tail -1
set -e
cd "$(dirname "$0")/../.."
if [ ! -d /tmp/cifar/train_images ]; then
  mkdir -p /tmp/cifar && unzip -q data/train_images.zip -d /tmp/cifar/train_images && unzip -q data/test_images.zip -d /tmp/cifar/test_images
fi
[ -L data/train_images ] || { rm -rf data/train_images data/test_images && ln -s /tmp/cifar/train_images data/train_images && ln -s /tmp/cifar/test_images data/test_images; }
echo "$(date +%H:%M) start kdcut (0.902 recipe + Cutout-16)" | tee -a pipeline.log
FULL=1 MIX=1 ARCH="64,128,256;4,4,3" WORKERS=10 SMOOTH=0.1 EPOCHS=400 SEED=41 T_KD=1 A_KD=1 KD_LOSS=kl CUTOUT=16 RESUME=1 TAG=_kdcut \
  TEACHERS="best_t2_400ep.pt,best_t2_400ep_s14.pt,best_t2_400ep_s15.pt,best_t2_600ep.pt" python -u train.py >> kdcut.log 2>&1
python tta_balance.py best_kdcut.pt sub_kdcut >> kdcut.log 2>&1
md5sum sub_kdcut_tta.csv best_kdcut.pt | tee -a pipeline.log
echo "$(date +%H:%M) done -> sub_kdcut_tta.csv" | tee -a pipeline.log
