# Run 21 (PC): the 0.902 recipe (run_kd4.ps1, 4 teachers, pure KL T=1, CutMix, 400 ep, all 50k) with ONE change:
# a 16x16 Cutout square on every image (CUTOUT=16) instead of the small random erase. Recht et al. 2018 measured Cutout
# at +0.1 on CIFAR-10 but more on shifted data. Judged only by the Kaggle score.
# Output: best_kdcut.pt, sub_kdcut_tta.csv (upload this), log kd_kdcut.log.
param([int]$Epochs = 400, [int]$Seed = 41, [string]$Tag = "_kdcut", [string]$Ck = "checkpoints")
Set-Location (Join-Path $PSScriptRoot "..\..")
Remove-Item Env:FOLD, Env:SAM, Env:TEACHER, Env:INIT, Env:LR, Env:AUG -ErrorAction SilentlyContinue
$env:EPOCHS = "$Epochs"; $env:FULL = "1"; $env:MIX = "1"; $env:WORKERS = "4"; $env:SMOOTH = "0.1"; $env:CUTOUT = "16"
$env:ARCH = "64,128,256;4,4,3"; $env:SEED = "$Seed"; $env:TAG = $Tag; $env:T_KD = "1"; $env:A_KD = "1"
$env:TEACHERS = (@("best_t2_400ep.pt", "best_t2_400ep_s14.pt", "best_t2_400ep_s15.pt", "best_t2_600ep.pt") | ForEach-Object { Join-Path $Ck $_ }) -join ","
py -3.11 -u train.py *> "kd$Tag.log"
py -3.11 tta_balance.py "best$Tag.pt" "sub$Tag" *>> "kd$Tag.log"
