#!/bin/bash
# Cross-fold distillation on 1x A100 (NYU HPC Cloud Bursting, run inside a Jupyter terminal).
# 3 teachers, each trained WITHOUT label smoothing on 2/3 of the 50k images (fold k held out),
# then one student whose soft target for each image comes only from the teacher that never saw it.
# Usage: bash hpc_pipeline.sh   (from the folder holding train.py and data/)   Progress: cat pipeline.log
set -e
cd "$(dirname "$0")/../.."  # project root (this script lives in scripts/hpc/)
if [ ! -d data/train_images ]; then
  (cd data && mkdir -p train_images test_images && unzip -q train_images.zip -d train_images && unzip -q test_images.zip -d test_images)
fi
echo "train images: $(ls data/train_images | wc -l), test images: $(ls data/test_images | wc -l)" | tee -a pipeline.log

export FULL=1 MIX=1 ARCH="64,128,256;4,4,3" WORKERS=10  # 12-core node; 4 workers left the A100 ~60% idle
for k in 0 1 2; do
  echo "$(date +%H:%M) start teacher fold $k" | tee -a pipeline.log
  FOLD=$k SMOOTH=0 SEED=$((20 + k)) TAG=_f$k python -u train.py > f$k.log 2>&1
done

echo "$(date +%H:%M) start cross-fold student" | tee -a pipeline.log
FOLD=-1 SMOOTH=0.1 T_KD=2 SEED=27 TAG=_cfstudent TEACHERS=best_f0.pt,best_f1.pt,best_f2.pt \
  python -u train.py > cfstudent.log 2>&1
echo "$(date +%H:%M) done -> submission_cfstudent.csv" | tee -a pipeline.log
