#!/bin/bash
# HPC: the 0.911 recipe with ResizeMix in place of CutMix (MIX_RESIZE=1; Qin et al. 2020, docs/research-plane-ship.md):
# on half the batches the whole partner image, shrunk to side 0.1-0.8 x 32, is pasted at a random spot, so the patch never is background-only.
# Everything else = 0.911: 4 teachers, pure KL T=1, TrivialAugment, smoothing 0.1, BLUR=0.2/1.0, ZOOM=0.3/0.6, all 50k, 400 ep, seed 41, 10-view TTA.
# ~1.5 h on 1x A100. Needs the current train.py (with MIX_RESIZE) and tta_balance.py. Resumable (RESUME=1): re-run to continue.
# Usage: nohup bash scripts/hpc/hpc_rmix.sh > /dev/null 2>&1 &   Progress: grep -a 'val acc' kd_rmix.log | tail -1
# Output: best_rmix.pt, sub_rmix_tta.csv (upload this; ignore the _hbal files), log kd_rmix.log.
set -e
cd "$(dirname "$0")/../.."  # project root (this script lives in scripts/hpc/)
if [ ! -d /tmp/cifar/train_images ]; then  # node-local disk: home is slow NFS
  mkdir -p /tmp/cifar && unzip -q data/train_images.zip -d /tmp/cifar/train_images && unzip -q data/test_images.zip -d /tmp/cifar/test_images
fi
[ -L data/train_images ] || { rm -rf data/train_images data/test_images && ln -s /tmp/cifar/train_images data/train_images && ln -s /tmp/cifar/test_images data/test_images; }
grep -q MIX_RESIZE train.py || { echo "train.py on HPC is old: upload the current one first"; exit 1; }
T=best_t2_400ep.pt,best_t2_400ep_s14.pt,best_t2_400ep_s15.pt,best_t2_600ep.pt
for f in ${T//,/ }; do [ -f "$f" ] || { echo "missing teacher $f"; exit 1; }; done
echo "$(date +%H:%M) start rmix (0.911 recipe, ResizeMix instead of CutMix, 400 ep)" | tee -a pipeline.log
FULL=1 MIX=1 MIX_RESIZE=1 ARCH="64,128,256;4,4,3" WORKERS=10 SMOOTH=0.1 EPOCHS=400 SEED=41 T_KD=1 A_KD=1 RESUME=1 TAG=_rmix \
  BLUR=0.2 BLUR_S=1.0 ZOOM=0.3 ZOOM_S=0.6 STRIDES=1,2,2 GRAY=0 TEACHERS=$T python -u train.py >> kd_rmix.log 2>&1
python tta_balance.py best_rmix.pt sub_rmix >> kd_rmix.log 2>&1
md5sum sub_rmix_tta.csv best_rmix.pt | tee -a pipeline.log
echo "$(date +%H:%M) done -> sub_rmix_tta.csv" | tee -a pipeline.log
