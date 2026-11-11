# ResNet on CIFAR-10 under 5M Parameters

**ECE-GY 7123 Deep Learning, Project 1 (Fall 2026), New York University**
Team **Jugaad Net**: Chetan Kodeboyina (ck3399), Shwetanshu Raj (sr8250), Ninad Rade (nr3263)
Repository: https://github.com/chetank03/resnet-cifar10-5m

We train a ResNet with **4.70M parameters** from scratch on the 50,000 CIFAR-10 training images (no pretrained weights, no external data) for a Kaggle challenge whose hidden 2,000-image test set comes from a newer, shifted collection of images.

| Result | |
|---|---|
| Best public Kaggle score | **0.911** (16th) |
| Private score of that submission (final ranking) | 0.876 (22nd) |
| Its accuracy on all 2,000 test images | 0.894, the best of our 33 submissions |
| Held-out accuracy (5k split, same recipe trained on 45k) | 98.64% |
| Trainable parameters | 4,697,162 (limit 5,000,000) |

**Start here:** [`ResNet_CIFAR10.ipynb`](ResNet_CIFAR10.ipynb) walks through the final model end to end, and [`report/report.pdf`](report/report.pdf) is the project report.

## Repository structure

```
├── ResNet_CIFAR10.ipynb      Main notebook: data, model, recipe, curves, inference, results, lessons
├── train.py                  Training script (all options are environment variables)
├── tta_balance.py            10-view test-time augmentation for a saved checkpoint
├── predict.py                Writes a Kaggle submission from any checkpoint (flip TTA)
├── requirements.txt
├── checkpoints/
│   ├── best_blurzoom.pt      Final model, trained on all 50k (EMA weights, 19 MB)
│   └── best_bz45.pt          Same recipe trained on 45k, for held-out analysis
├── results/
│   └── submission_final_0.911.csv   The submitted predictions (public score 0.911)
├── logs/
│   └── train_log_final.txt   Per-epoch training log of the final model
├── figures/                  Augmentation samples and training curves
├── report/                   Report (PDF + LaTeX) and the scripts that make its figures
├── docs/
│   └── EXPERIMENTS.md        Every run and submission, with scores and what changed
├── scripts/
│   ├── pc/                   PowerShell run scripts (RTX 5070 Ti); run_blur.ps1 reproduces 0.911
│   └── hpc/                  Bash/SLURM scripts for NYU HPC (A100)
└── data/                     Not included: download from Kaggle (see below)
```

## Approach

- **Model:** basic-block ResNet (He et al., 2016), channels (64, 128, 256) with (4, 4, 3) blocks per stage, global average pooling and a linear classifier.
- **Training:** SGD with Nesterov momentum, learning rate 0.1 with cosine decay over 400 epochs, weight decay 5e-4, label smoothing 0.1, EMA of weights.
- **Augmentation:** crop + flip, TrivialAugment, small random erasing, CutMix on half the batches, plus two shift-targeted augmentations: Gaussian blur (20% of images) and zoom-out (30%).
- **Distillation:** the model matches the averaged predictions of four same-size teachers (pure KL, temperature 1), following Beyer et al. (2022).
- **Inference:** softmax averaged over 10 views (image and mirror, five 2-pixel shifts each).

## Key findings

1. **Most of the gain came from the basics.** All the data, the full parameter budget, CutMix and 400 epochs took the score from 0.874 to 0.899.
2. **Blur and zoom-out gave the best public score (0.911).** We chose them by measuring how models fail on degraded images. Averaged over seeds on the full test set they are worth about 0.004, which is within noise.
3. **Pushing working ideas further did not help.** Longer training (600–1,200 epochs), bigger teachers (36.7M parameters) and stronger augmentation all scored lower.
4. **Held-out accuracy did not predict the leaderboard.** It was saturated near 98.6% for every distilled variant.
5. **Seed noise is as large as most effects.** The exact 0.911 recipe scored 0.895 and 0.898 with seeds 42 and 43.
6. **Neither half of the test set could rank models this close.** Our two selected submissions (the best public scores) both scored 0.876 on the private half, and seven unselected submissions scored higher. But each half has a standard error of about 0.010, and on all 2,000 images our selected submission is the most accurate of the 33.
7. **Distillation added almost nothing.** Averaged over seeds on the full test set it is worth about 0.001. The basics account for 0.021 of the 0.025 total gain and are the only part clearly larger than the noise.

The full list of experiments is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Reproducing

```bash
pip install -r requirements.txt
```

1. Download the competition data from Kaggle and unzip it into `data/` so it contains `train_images/`, `test_images/`, `train_labels.csv` and `test.csv`.
2. Open `ResNet_CIFAR10.ipynb` and run all cells. It loads `checkpoints/best_blurzoom.pt` and rebuilds the submission (identical to `results/submission_final_0.911.csv`).
3. To regenerate a submission from the command line:
   ```bash
   python tta_balance.py checkpoints/best_blurzoom.pt sub   # writes sub_tta.csv
   ```
4. To retrain the final model (about 75 minutes on an RTX 5070 Ti), first train the four teachers (`scripts/pc/run400.ps1`, `run400_seed.ps1`, `run600.ps1`) into `checkpoints/`, then run `scripts/pc/run_blur.ps1` or the command in the notebook.

Training is seeded, but cuDNN and data loading are not fully deterministic, so retrained models differ slightly. Seeds alone moved our public score by up to 0.02.

## Rules followed

- One model, under 5M trainable parameters (`train.py` asserts this), trained from scratch.
- No external training data or pretrained weights.
- Local evaluation used only a 5,000-image held-out split of the training images.

## References

He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016) · Yun et al., *CutMix* (ICCV 2019) · Müller & Hutter, *TrivialAugment* (ICCV 2021) · Hinton et al., *Distilling the Knowledge in a Neural Network* (2015) · Beyer et al., *Knowledge Distillation: A Good Teacher Is Patient and Consistent* (CVPR 2022) · Recht et al., *Do CIFAR-10 Classifiers Generalize to CIFAR-10?* (2018)
