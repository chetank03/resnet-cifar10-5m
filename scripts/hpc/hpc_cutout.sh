#!/bin/bash
# Run 21 (HPC): the 0.902 recipe (4 same-size teachers, pure KL T=1, CutMix, 400 ep, all 50k) with ONE change: a 16x16
# Cutout square on every image (CUTOUT=16) instead of the small random erase. Recht et al. 2018 measured Cutout at +0.1 on
# CIFAR-10 but more on shifted test data. ~1.5 h on 1x A100. Needs the current train.py.
# Resumable (RESUME=1). Usage: nohup bash scripts/hpc/hpc_cutout.sh > /dev/null 2>&1 &   Progress: grep -a 'val acc' kd_kdcut.log | tail -1
# Output: best_kdcut.pt, sub_kdcut_tta.csv (upload this; ignore the _hbal files), log kd_kdcut.log.
set -e
cd "$(dirname "$0")/../.."  # project root (this script lives in scripts/hpc/)
if [ ! -d /tmp/cifar/train_images ]; then  # node-local disk: home is slow NFS
  mkdir -p /tmp/cifar && unzip -q data/train_images.zip -d /tmp/cifar/train_images && unzip -q data/test_images.zip -d /tmp/cifar/test_images
fi
[ -L data/train_images ] || { rm -rf data/train_images data/test_images && ln -s /tmp/cifar/train_images data/train_images && ln -s /tmp/cifar/test_images data/test_images; }
grep -q CUTOUT train.py || { echo "train.py on HPC is old: upload the current one first"; exit 1; }
echo "$(date +%H:%M) start kdcut student (Cutout 16, 400 ep)" | tee -a pipeline.log
FULL=1 MIX=1 CUTOUT=16 ARCH="64,128,256;4,4,3" WORKERS=10 EPOCHS=400 SEED=41 RESUME=1 TAG=_kdcut T_KD=1 A_KD=1 \
  TEACHERS=best_t2_400ep.pt,best_t2_400ep_s14.pt,best_t2_400ep_s15.pt,best_t2_600ep.pt python -u train.py >> kd_kdcut.log 2>&1
python tta_balance.py best_kdcut.pt sub_kdcut >> kd_kdcut.log 2>&1
md5sum sub_kdcut_tta.csv best_kdcut.pt | tee -a pipeline.log
echo "$(date +%H:%M) done -> sub_kdcut_tta.csv" | tee -a pipeline.log
