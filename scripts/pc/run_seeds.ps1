# Runs 22+: the 0.902 recipe (run_kd4.ps1: 4 same-size teachers, pure KL T=1, CutMix, 400 ep, all 50k) repeated with
# new seeds, one after another (one GPU). Each seed is an independent draw of the same model; only the Kaggle public
# score tells them apart, and the differences are noise (+-0.9 pts on 1,000 images). ~2.5 h per seed on the 5070 Ti.
# Output per seed: best_s<seed>.pt, sub_s<seed>_tta.csv (upload this; ignore the _hbal files), log kd_s<seed>.log.
# Resumable: a seed whose sub_s<seed>_tta.csv already exists is skipped, so re-running the script continues the list.
param([int[]]$Seeds = @(201, 202, 203, 204, 205), [int]$Epochs = 400, [string]$Ck = "checkpoints")
Set-Location (Join-Path $PSScriptRoot "..\..")
Remove-Item Env:FOLD, Env:SAM, Env:TEACHER, Env:INIT, Env:LR, Env:AUG, Env:CUTOUT -ErrorAction SilentlyContinue
$env:EPOCHS = "$Epochs"; $env:FULL = "1"; $env:MIX = "1"; $env:WORKERS = "4"; $env:SMOOTH = "0.1"; $env:RESUME = "1"
$env:ARCH = "64,128,256;4,4,3"; $env:T_KD = "1"; $env:A_KD = "1"
$env:TEACHERS = (@("best_t2_400ep.pt", "best_t2_400ep_s14.pt", "best_t2_400ep_s15.pt", "best_t2_600ep.pt") | ForEach-Object { Join-Path $Ck $_ }) -join ","
foreach ($s in $Seeds) {
    $tag = "_s$s"
    if (Test-Path "sub${tag}_tta.csv") { continue }
    $env:SEED = "$s"; $env:TAG = $tag
    "$(Get-Date -Format HH:mm) start seed $s" | Add-Content pipeline.log
    py -3.11 -u train.py *> "kd$tag.log"
    py -3.11 tta_balance.py "best$tag.pt" "sub$tag" *>> "kd$tag.log"
    "$(Get-Date -Format HH:mm) done seed $s -> sub${tag}_tta.csv" | Add-Content pipeline.log
}
