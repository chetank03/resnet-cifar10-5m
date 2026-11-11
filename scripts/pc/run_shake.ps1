# Run 24 (PC): Shake-Shake 26 2x40d student (Gastaldi 2017; 4,578,610 params) distilled with the 0.902 recipe
# (4 same-size ResNet teachers, pure KL T=1, CutMix, TrivialAugment, all 50k, EMA, 10-view TTA). Paper schedule:
# lr 0.2, wd 1e-4, cosine, 1800 epochs (~28 s/epoch on the 5070 Ti -> ~14 h). Resumable: re-run this script (RESUME=1).
# Output: best_ss40.pt, sub_ss40_tta.csv (upload this), log kd_ss40.log. Progress: scripts\pc\watch.ps1 or tail pipeline.log.
param([int]$Epochs = 1800, [int]$Seed = 1, [string]$Tag = "_ss40", [string]$Ck = "checkpoints")
Set-Location (Join-Path $PSScriptRoot "..\..")
Remove-Item Env:FOLD, Env:SAM, Env:TEACHER, Env:INIT, Env:AUG, Env:CUTOUT, Env:SE, Env:DROP_PATH -ErrorAction SilentlyContinue
$env:EPOCHS = "$Epochs"; $env:FULL = "1"; $env:MIX = "1"; $env:WORKERS = "4"; $env:SMOOTH = "0.1"; $env:RESUME = "1"
$env:SHAKE = "1"; $env:ARCH = "40,80,160;4,4,4"; $env:LR = "0.2"; $env:WD = "1e-4"
$env:SEED = "$Seed"; $env:TAG = $Tag; $env:T_KD = "1"; $env:A_KD = "1"
$env:TEACHERS = (@("best_t2_400ep.pt", "best_t2_400ep_s14.pt", "best_t2_400ep_s15.pt", "best_t2_600ep.pt") | ForEach-Object { Join-Path $Ck $_ }) -join ","
"$(Get-Date -Format HH:mm) start shake-shake 2x40d student $Tag ($Epochs ep, seed $Seed)" | Add-Content pipeline.log
py -3.11 -u train.py *>> "kd$Tag.log"
py -3.11 tta_balance.py "best$Tag.pt" "sub$Tag" *>> "kd$Tag.log"
"$(Get-Date -Format HH:mm) done $Tag -> sub${Tag}_tta.csv" | Add-Content pipeline.log
