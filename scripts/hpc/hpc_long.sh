#!/bin/bash
# Run 13 (HPC): "patient" distillation, 1200 epochs, 9 teachers (runs 6, 6b, 6c, 10, 11, 12), pure KL at T=1, then 10-view TTA.
# Resumable: RESUME=1 saves state_long.pt every epoch; re-running this script in a new session continues where it stopped.
# Usage: nohup bash hpc_long.sh > /dev/null 2>&1 &   (from ~/ResNet-CIFAR10)   Progress: grep -a -o 'ep *[0-9]*/1200 *loss[^\r]*' long.log | tail -1
set -e
cd "$(dirname "$0")/../.."  # project root (this script lives in scripts/hpc/)
if [ ! -d /tmp/cifar/train_images ]; then
  mkdir -p /tmp/cifar && unzip -q data/train_images.zip -d /tmp/cifar/train_images && unzip -q data/test_images.zip -d /tmp/cifar/test_images
fi
[ -L data/train_images ] || { rm -rf data/train_images data/test_images && ln -s /tmp/cifar/train_images data/train_images && ln -s /tmp/cifar/test_images data/test_images; }
T=""
for f in best_t2_400ep.pt best_t2_400ep_s14.pt best_t2_400ep_s15.pt best_t2_600ep.pt best_kd4.pt best_se_s16.pt best_se_s17.pt best_final.pt best_final2.pt; do
  [ -f "$f" ] && T="$T${T:+,}$f"
done
echo "$(date +%H:%M) start long (1200 ep), teachers: $T" | tee -a pipeline.log
FULL=1 MIX=1 ARCH="64,128,256;4,4,3" WORKERS=10 EPOCHS=1200 SEED=51 TEACHERS="$T" T_KD=1 A_KD=1 RESUME=1 TAG=_long python -u train.py >> long.log 2>&1
[ -f submission_long.csv ] || python predict.py best_long.pt submission_long.csv >> long.log 2>&1
python tta_balance.py best_long.pt sublong >> long.log 2>&1
md5sum sublong_tta.csv submission_long.csv best_long.pt | tee -a pipeline.log
echo "$(date +%H:%M) done -> sublong_tta.csv" | tee -a pipeline.log
