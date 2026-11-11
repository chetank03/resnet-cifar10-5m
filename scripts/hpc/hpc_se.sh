#!/bin/bash
# Run 11 (HPC): SE + stochastic depth teacher, t2 recipe, 400 epochs, seed 17, on 1x A100. ~90 min.
# Usage: nohup bash hpc_se.sh > /dev/null 2>&1 &   (from ~/ResNet-CIFAR10)   Progress: grep -a 'val acc' se_s17.log | tail -1
set -e
cd "$(dirname "$0")/../.."  # project root (this script lives in scripts/hpc/)
if [ ! -d /tmp/cifar/train_images ]; then  # node-local disk: home is slow NFS
  mkdir -p /tmp/cifar && unzip -q data/train_images.zip -d /tmp/cifar/train_images && unzip -q data/test_images.zip -d /tmp/cifar/test_images
fi
rm -rf data/train_images data/test_images && ln -s /tmp/cifar/train_images data/train_images && ln -s /tmp/cifar/test_images data/test_images
echo "train images: $(ls data/train_images/ | wc -l), test images: $(ls data/test_images/ | wc -l)"
echo "$(date +%H:%M) start se_s17" | tee -a pipeline.log
FULL=1 MIX=1 ARCH="64,128,256;4,4,3" WORKERS=10 EPOCHS=400 SEED=17 SE=1 DROP_PATH=0.1 TAG=_se_s17 python -u train.py > se_s17.log 2>&1
[ -f submission_se_s17.csv ] || python predict.py best_se_s17.pt submission_se_s17.csv >> se_s17.log 2>&1
echo "$(date +%H:%M) done -> submission_se_s17.csv" | tee -a pipeline.log
