# Run 18 (PC): the 0.902 recipe (run_kd4.ps1: 4 same-size teachers, pure KL T=1, CutMix, 400 ep) but FULL=0, so the
# 5k held-out val is honest. It is the base model for the soup experiment (run_soup.ps1) and the number the soup must beat.
# Teachers live in the Desktop working copy. Output: best_kd45.pt, log kd_kd45.log.
param([int]$Epochs = 400, [int]$Seed = 41, [string]$Tag = "_kd45", [string]$Ck = "checkpoints")
Set-Location (Join-Path $PSScriptRoot "..\..")
Remove-Item Env:FOLD, Env:SAM, Env:TEACHER, Env:INIT, Env:LR -ErrorAction SilentlyContinue
$env:EPOCHS = "$Epochs"; $env:FULL = "0"; $env:MIX = "1"; $env:WORKERS = "4"; $env:SMOOTH = "0.1"
$env:ARCH = "64,128,256;4,4,3"; $env:SEED = "$Seed"; $env:TAG = $Tag; $env:T_KD = "1"; $env:A_KD = "1"
$env:TEACHERS = (@("best_t2_400ep.pt", "best_t2_400ep_s14.pt", "best_t2_400ep_s15.pt", "best_t2_600ep.pt") | ForEach-Object { Join-Path $Ck $_ }) -join ","
py -3.11 -u train.py *> "kd$Tag.log"
