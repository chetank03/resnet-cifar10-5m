#!/bin/bash
# Run 12 (HPC): final student distilled from every full-data teacher present (runs 6, 6b, 6c, 10, 11), pure KL at T=1, 400 epochs, then 10-view TTA.
# Usage: nohup bash hpc_final.sh > /dev/null 2>&1 &   (from ~/ResNet-CIFAR10, after hpc_se.sh)   Progress: grep -a 'val acc' final.log | tail -1
set -e
cd "$(dirname "$0")/../.."  # project root (this script lives in scripts/hpc/)
if [ ! -d /tmp/cifar/train_images ]; then
  mkdir -p /tmp/cifar && unzip -q data/train_images.zip -d /tmp/cifar/train_images && unzip -q data/test_images.zip -d /tmp/cifar/test_images
fi
rm -rf data/train_images data/test_images && ln -s /tmp/cifar/train_images data/train_images && ln -s /tmp/cifar/test_images data/test_images
T=""
for f in best_t2_400ep.pt best_t2_400ep_s14.pt best_t2_400ep_s15.pt best_t2_600ep.pt best_kd4.pt best_se_s16.pt best_se_s17.pt; do
  [ -f "$f" ] && T="$T${T:+,}$f"
done
echo "$(date +%H:%M) start final, teachers: $T" | tee -a pipeline.log
FULL=1 MIX=1 ARCH="64,128,256;4,4,3" WORKERS=10 EPOCHS=400 SEED=43 TEACHERS="$T" T_KD=1 A_KD=1 TAG=_final python -u train.py > final.log 2>&1
[ -f submission_final.csv ] || python predict.py best_final.pt submission_final.csv >> final.log 2>&1
python tta_balance.py best_final.pt subfinal >> final.log 2>&1
md5sum subfinal_tta.csv submission_final.csv best_final.pt | tee -a pipeline.log
echo "$(date +%H:%M) done -> subfinal_tta.csv" | tee -a pipeline.log
