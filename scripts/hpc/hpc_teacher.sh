#!/bin/bash
# Run 16b (HPC): big distillation teacher on all 50k, WRN-28-10-shaped (160,320,640)x(4,4,4), ~36.7M params (TEACHER=1 lifts the 5M cap; never submitted).
# Same recipe as t2 (CutMix, TrivialAugment, EMA, smoothing 0.1), 200 epochs, on 1x A100 (~3 h). The PC trains the 45k/5k twin (scripts/pc/run_teacher.ps1) for an honest val number.
# Resumable: RESUME=1 saves state_big50.pt every epoch (~440 MB); re-running this script in a new session continues where it stopped.
# Usage: nohup bash scripts/hpc/hpc_teacher.sh > /dev/null 2>&1 &   (from ~/ResNet-CIFAR10)   Progress: grep -a 'val acc' teacher_big50.log | tail -1
set -e
cd "$(dirname "$0")/../.."  # project root (this script lives in scripts/hpc/)
if [ ! -d /tmp/cifar/train_images ]; then  # node-local disk: home is slow NFS
  mkdir -p /tmp/cifar && unzip -q data/train_images.zip -d /tmp/cifar/train_images && unzip -q data/test_images.zip -d /tmp/cifar/test_images
fi
[ -L data/train_images ] || { rm -rf data/train_images data/test_images && ln -s /tmp/cifar/train_images data/train_images && ln -s /tmp/cifar/test_images data/test_images; }
echo "$(date +%H:%M) start big50 teacher (200 ep)" | tee -a pipeline.log
TEACHER=1 FULL=1 MIX=1 ARCH="160,320,640;4,4,4" WORKERS=10 EPOCHS=200 SEED=22 RESUME=1 TAG=_big50 python -u train.py >> teacher_big50.log 2>&1
md5sum best_big50.pt | tee -a pipeline.log
echo "$(date +%H:%M) done -> best_big50.pt" | tee -a pipeline.log
