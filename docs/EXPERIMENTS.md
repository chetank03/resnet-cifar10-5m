# Experiment log: everything tried

**Task:** CIFAR-10, one model, at most 5M parameters, trained from scratch on the course's 50k training PNGs.
- No external data.
- The hidden test set is 2,000 images from a newer, shifted collection of images. Its public score covers about half of them, with noise of about ±0.01.

**Best:** **0.911** public (`submissions/best_0.911_blurzoom/`). Runs 1–17 are also in the README table, and `docs/research-0.92.md` has the literature review.

**Rules followed throughout:**
- The hidden test images are never used for training or model selection. They aren't hand-labelled either.
- Local evaluation uses only a 5k held-out split of the training images.

## 1. Public scores (every upload)

| Score | Run | What changed vs the line it came from |
|---|---|---|
| **0.911** | `_blurzoom` (run 26) | kd4 recipe + Gaussian blur on 20% (σ 0.3–1.0) + zoom-out on 30% (60–100% size). **Best.** |
| 0.902 | `_bz800` (HPC) | 0.911 recipe trained 800 epochs instead of 400. Lower public score than 400 epochs |
| 0.902 | kd4 (run 10) | Student distilled from 3 same-size teachers (pure KL, T=1), CutMix, TrivialAugment, 400 ep, 10-view TTA |
| 0.899 | t2 400 ep (run 6) | No distillation, 400 epochs (seeds 14/15: 0.894/0.899) |
| 0.899 | `_kdaug` | AugMix instead of TrivialAugment |
| 0.899 | run 10b | Second distillation round (student added as a teacher) |
| 0.898 | `_blurzoom2` | Stronger blur (σ ≤ 1.2, 30%) + stronger zoom (scale ≥ 0.5, 40%): too strong |
| 0.898 | run 13 | 1200 epochs, 9 teachers |
| 0.897 | `_rmix` (HPC) | 0.911 recipe with ResizeMix (whole partner image shrunk to 10–80% and pasted) instead of CutMix |
| 0.897 | run 6b | 600 epochs, no distillation |
| 0.895 | `_nomix` | 0.911 recipe **without CutMix** |
| 0.895 | `_s201` | kd4 recipe, different seed |
| 0.895 | run 8 | SAM (ρ 0.05) |
| 0.894 | run 14 | MSE on logits instead of KL distillation |
| 0.894 | `_kdcut` | Cutout 16×16 instead of the small random erase |
| 0.893 | run 17 | Distilled from one 36.7M WRN-sized teacher |
| 0.892 | run 15 | Wider, shallower student (128,256,320)×(2,2,1) |
| 0.891 | `_bteach` | Better teachers (0.911/0.902/0.898 students + t2_600ep) |
| 0.89x | `_delay` | Delayed downsampling (strides 1,2,1) |
| 0.89 | `_bz42`, `_bz43` | **Exact 0.911 recipe, seeds 42 and 43.** Shows 0.911 (seed 41) was a lucky draw; the recipe's typical level is about 0.89–0.90 |
| 0.89 | `_ss40b` | Shake-Shake 26 2×40d (4.58M), 1800 ep, no blur/zoom (HPC) |
| 0.889 | t2 200 ep (run 3) | All 50k + CutMix + deeper 4-4-3 net |
| 0.888 | run 5 | First distilled student (T=4) |
| 0.875 | run 7 | Cross-fold distillation |
| 0.874 | run 1 | Baseline (64,128,256)×(3,3,3), 45k, 200 ep |
| uploaded | `_gray45` | 0.911 recipe + 10% grayscale, 45k (user: no improvement; exact score not recorded here) |

## 2. Local evaluation (held-out 5k only)

All local numbers use the first 5,000 training images after a seed-0 shuffle, with models trained on the other 45,000. Single-view accuracy except the TTA column.

| Model (45k) | Held-out, 10-view TTA | clean | blur σ1.2 | downscale to 16 px | 16-px object |
|---|---|---|---|---|---|
| Baseline (3,3,3), 200 ep | 97.22 | — | — | — | — |
| kd4 recipe (no blur/zoom), `_kd45` | 98.56 | 98.42 | 83.72 | 80.64 | 79.08 |
| **0.911 recipe, `_bz45`** | **98.64** | 98.44 | 86.92 | 83.88 | 81.88 |
| + 10% grayscale, `_gray45` | 98.70 | 98.52 | 87.14 | 84.02 | 82.22 |
| 36.7M teacher, `_big45` (200 ep) | 97.84 | — | — | — | — |

- Blur + zoom-out raises accuracy under blur by 3.2 points and on 16-px objects by 2.8, with clean accuracy unchanged.
- Distilled variants differ by at most 0.15 held-out, while their full-data versions differ by up to 0.02 on Kaggle: held-out accuracy is saturated.
- `_bz45` makes 68 held-out errors; cat↔dog is 25 of them.

## 3. What we learned

1. **What helped on the hidden set:**
   - more data and a deeper net (+1.5)
   - 200 → 400 epochs (+1.0)
   - ensemble distillation (+0.3)
   - blur + zoom-out (+0.9, but within seed noise: the same recipe scored 0.89 with seeds 42 and 43)
2. **What didn't help:**
   - longer training (600, 800, 1200 epochs)
   - SAM, MSE distillation, second-round distillation, cross-fold distillation
   - bigger or better teachers
   - Cutout, AugMix, ResizeMix
   - wider/shallower or Shake-Shake students
   - delayed downsampling
   - stronger blur/zoom
   - grayscale
3. **No local test predicts the hidden score.** Held-out images come from the training distribution, so they cannot see the shift. Only uploads settle it, and they are noisy (±0.01 from the public split, about ±0.01 more from the seed).
4. **Seed noise is as large as most effects.** Two seeds of the same recipe disagree on 72–80 of the 2,000 hidden images, more than 800 epochs (58) or ResizeMix (68) changed.
5. **The loss curve's shape comes from the cosine schedule.** The loss falls until about epoch 375, then flattens as the learning rate reaches 0. Most held-out gain comes in epochs 200–400.

## 4. Later runs

| Run | What | Public |
|---|---|---|
| `_bz800` | 0.911 recipe, 800 epochs | 0.902 |
| `_rmix` | 0.911 recipe, ResizeMix instead of CutMix | 0.897 |
| `_bz42`, `_bz43` | exact 0.911 recipe, seeds 42 and 43 | 0.89, 0.89 |
| `_zoomin` | 0.911 recipe + zoom-in crops on 30% of images | not uploaded |
| `_nt42/43/44`, `_noteach` | 0.911 recipe without teachers | not uploaded |
| `_bzkd` | 0.911 recipe distilled from the teacher-free models | not uploaded |

## 5. Where things are

- **Root:**
  - code: `train.py`, `tta_balance.py`, `predict.py`, `soup.py`
  - `results.ipynb`
  - the running job's files: `*_zoomin*`, `pipeline.log`
  - New runs write their outputs to the root. Move them into the folders below when they finish.
- `checkpoints/`: every finished `best_*.pt`.
  - `checkpoints/state/` holds resume states, only needed to continue a stopped run.
  - Old scripts that load a checkpoint by bare name (`run_soup.ps1 -Init best_kd45.pt`, `run_kd_big.ps1 -Teacher best_big45.pt`) need the `checkpoints\` prefix or a copy in the root.
- `submissions/`: every prediction file.
  - `*_tta.csv` is the file to upload.
  - `*_hbal` / `*_ttahbal` force 200 per class and were never better.
  - `uploaded/` is named by public score.
  - `best_0.911_blurzoom/` and `best_0.902_kd4/` hold the best entries with their code.
- `logs/runs/`: one training log per run. They are UTF-16; read with `iconv -f UTF-16LE -t UTF-8`.
- `logs/figures/`: all analysis figures.
  - disagreement grids (`*disagree*`)
  - error grids
  - augmentation samples
  - loss curves
  - research figures
- `scripts/pc/`, `scripts/hpc/`: run scripts. `scripts/pc/run_blur.ps1` reproduces 0.911 and has switches:
  - `-Full`, `-Gray`, `-Mix`, `-Erase`, `-ZoomIn`, `-NoTeachers`, `-Epochs`, `-Tag`
- **Teachers** for the 0.911 recipe: `checkpoints/best_t2_400ep*.pt`, `best_t2_600ep.pt`.

**`train.py` switches added in this round (all default off):**
- `GRAY` (grayscale fraction)
- `ERASE=0` (no random erase)
- `ZOOMIN` (zoom-in fraction)

`from_ckpt` now defaults to one stride per stage, so old 4-stage checkpoints load.
