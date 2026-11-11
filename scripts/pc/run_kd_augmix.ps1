# Run 20 (PC): the 0.902 recipe (run_kd4.ps1, 4 teachers, pure KL T=1, CutMix, 400 ep, all 50k) with ONE change:
# AugMix instead of TrivialAugment (AUG=augmix). Val 0.972 vs public 0.90 says the test images are shifted, and
# AugMix is the standard augmentation for robustness to such shifts. Judged only by the Kaggle score (no honest val on 50k).
# Output: best_kdaug.pt, subkdaug_tta.csv (upload this), log kd_kdaug.log.
param([int]$Epochs = 400, [int]$Seed = 41, [string]$Tag = "_kdaug", [string]$Ck = "checkpoints")
Set-Location (Join-Path $PSScriptRoot "..\..")
Remove-Item Env:FOLD, Env:SAM, Env:TEACHER, Env:INIT, Env:LR -ErrorAction SilentlyContinue
$env:EPOCHS = "$Epochs"; $env:FULL = "1"; $env:MIX = "1"; $env:WORKERS = "4"; $env:SMOOTH = "0.1"; $env:AUG = "augmix"
$env:ARCH = "64,128,256;4,4,3"; $env:SEED = "$Seed"; $env:TAG = $Tag; $env:T_KD = "1"; $env:A_KD = "1"
$env:TEACHERS = (@("best_t2_400ep.pt", "best_t2_400ep_s14.pt", "best_t2_400ep_s15.pt", "best_t2_600ep.pt") | ForEach-Object { Join-Path $Ck $_ }) -join ","
py -3.11 -u train.py *> "kd$Tag.log"
py -3.11 tta_balance.py "best$Tag.pt" "sub$Tag" *>> "kd$Tag.log"
