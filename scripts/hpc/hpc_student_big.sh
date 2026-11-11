#!/bin/bash
# Run 17 (HPC): 4.7M student distilled from the 36.7M big teacher (best_big50.pt), pure KL at T=1, 400 epochs, then 10-view TTA.
# Usage: nohup bash scripts/hpc/hpc_student_big.sh > /dev/null 2>&1 &   (from ~/ResNet-CIFAR10)   Resumable (RESUME=1).
set -e
cd "$(dirname "$0")/../.."
[ -L data/train_images ] || { rm -rf data/train_images data/test_images && ln -s /tmp/cifar/train_images data/train_images && ln -s /tmp/cifar/test_images data/test_images; }
echo "$(date +%H:%M) start bigkd student (teacher best_big50.pt)" | tee -a pipeline.log
FULL=1 MIX=1 ARCH="64,128,256;4,4,3" WORKERS=10 EPOCHS=400 SEED=81 TEACHERS=best_big50.pt T_KD=1 A_KD=1 KD_LOSS=kl RESUME=1 TAG=_bigkd python -u train.py >> bigkd.log 2>&1
python tta_balance.py best_bigkd.pt subbigkd >> bigkd.log 2>&1
md5sum subbigkd_tta.csv best_bigkd.pt | tee -a pipeline.log
echo "$(date +%H:%M) done -> subbigkd_tta.csv" | tee -a pipeline.log
