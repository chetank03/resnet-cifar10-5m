#!/bin/bash
# Run 8: the 0.899 recipe (t2, 400 epochs) trained with SAM (rho=0.05) on 1x A100. ~75 min.
# Usage: bash hpc_sam.sh   (from ~/ResNet-CIFAR10)   Progress: grep -a 'val acc' sam400.log | tail -1
set -e
cd "$(dirname "$0")/../.."  # project root (this script lives in scripts/hpc/)
if [ ! -d /tmp/cifar/train_images ]; then  # node-local disk: home is slow NFS
  mkdir -p /tmp/cifar && unzip -q data/train_images.zip -d /tmp/cifar/train_images && unzip -q data/test_images.zip -d /tmp/cifar/test_images
fi
rm -rf data/train_images data/test_images && ln -s /tmp/cifar/train_images data/train_images && ln -s /tmp/cifar/test_images data/test_images
echo "train images: $(ls data/train_images/ | wc -l), test images: $(ls data/test_images/ | wc -l)"
echo "$(date +%H:%M) start sam400" | tee -a pipeline.log
FULL=1 MIX=1 ARCH="64,128,256;4,4,3" WORKERS=10 EPOCHS=400 SEED=13 SAM=0.05 TAG=_sam400 python -u train.py > sam400.log 2>&1
echo "$(date +%H:%M) done -> submission_sam400.csv" | tee -a pipeline.log
