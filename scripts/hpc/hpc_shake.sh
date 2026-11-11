#!/bin/bash
# Run 25 (HPC): Shake-Shake 26 2x40d student (4,578,610 params), seed 2, same recipe as scripts/pc/run_shake.ps1
# (4 teachers, pure KL T=1, CutMix, all 50k, lr 0.2, wd 1e-4, cosine 1800 ep). ~10 s/epoch on an A100 -> ~5 h.
# Resumable (RESUME=1): if the session ends, just launch this script again. Needs the current train.py + tta_balance.py.
# Usage: nohup bash scripts/hpc/hpc_shake.sh > /dev/null 2>&1 &   Progress: grep -a 'val acc' kd_ss40b.log | tail -1
# Second seed in parallel: SEED=3 TAG=_ss40c nohup bash scripts/hpc/hpc_shake.sh > /dev/null 2>&1 &   (log kd_ss40c.log, output sub_ss40c_tta.csv)
# Output: best_ss40b.pt, sub_ss40b_tta.csv (upload this; ignore the _hbal files), log kd_ss40b.log.
set -e
cd "$(dirname "$0")/../.."  # project root (this script lives in scripts/hpc/)
if [ ! -d /tmp/cifar/train_images ]; then  # node-local disk: home is slow NFS
  mkdir -p /tmp/cifar && unzip -q data/train_images.zip -d /tmp/cifar/train_images && unzip -q data/test_images.zip -d /tmp/cifar/test_images
fi
[ -L data/train_images ] || { rm -rf data/train_images data/test_images && ln -s /tmp/cifar/train_images data/train_images && ln -s /tmp/cifar/test_images data/test_images; }
grep -q SHAKE train.py || { echo "train.py on HPC is old: upload the current train.py and tta_balance.py first"; exit 1; }
SEED=${SEED:-2}; TAG=${TAG:-_ss40b}  # override: SEED=3 TAG=_ss40c bash scripts/hpc/hpc_shake.sh
echo "$(date +%H:%M) start shake-shake 2x40d student $TAG (1800 ep, seed $SEED)" | tee -a pipeline.log
FULL=1 MIX=1 SHAKE=1 ARCH="40,80,160;4,4,4" LR=0.2 WD=1e-4 WORKERS=10 EPOCHS=1800 SEED=$SEED RESUME=1 TAG=$TAG T_KD=1 A_KD=1 SMOOTH=0.1 \
  TEACHERS=best_t2_400ep.pt,best_t2_400ep_s14.pt,best_t2_400ep_s15.pt,best_t2_600ep.pt python -u train.py >> kd$TAG.log 2>&1
python tta_balance.py best$TAG.pt sub$TAG >> kd$TAG.log 2>&1
md5sum sub${TAG}_tta.csv best$TAG.pt | tee -a pipeline.log
echo "$(date +%H:%M) done -> sub${TAG}_tta.csv" | tee -a pipeline.log
