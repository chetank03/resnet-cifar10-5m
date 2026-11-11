# Run 26 (PC): the 0.902 recipe (best_0.902/run_kd4.ps1: 4 same-size teachers, pure KL T=1, CutMix, TrivialAugment, all 50k,
# 400 ep, seed 41, EMA, 10-view TTA) with two low-detail changes: BLUR=0.2 Gaussian-blurs 20% of training images (sigma 0.3-1.0) and
# ZOOM=0.3 zooms 30% out to 60-100% size on a mean-colour canvas (small objects).
# Held-out checks: the recipe loses 3-18 points under moderate blur and 7-19 points at 16-20 px objects (cat/dog worst). Resumable: re-run this script.
# Output: best_blurzoom.pt, sub_blurzoom_tta.csv (upload this), log kd_blurzoom.log.
param([int]$Epochs = 400, [int]$Seed = 41, [string]$Blur = "0.2", [string]$Zoom = "0.3", [string]$BlurS = "1.0", [string]$ZoomS = "0.6", [string]$Tag = "_blurzoom", [string]$Strides = "1,2,2", [string]$Teachers = "", [string]$Full = "1", [string]$Gray = "0", [string]$Mix = "1", [string]$Erase = "1", [string]$ZoomIn = "0",[switch]$Force, [switch]$NoTeachers, [string]$Ck = "checkpoints")
Set-Location (Join-Path $PSScriptRoot "..\..")
if ((Test-Path "skip$Tag") -and -not $Force) { "skip $Tag (already started separately)"; return }  # lets a queued run be started early in parallel
Remove-Item Env:FOLD, Env:SAM, Env:TEACHER, Env:INIT, Env:AUG, Env:CUTOUT, Env:SE, Env:DROP_PATH, Env:SHAKE, Env:LR, Env:WD, Env:KD_LOSS -ErrorAction SilentlyContinue
$env:EPOCHS = "$Epochs"; $env:FULL = $Full; $env:MIX = $Mix; $env:ERASE = $Erase; $env:ZOOMIN = $ZoomIn; $env:WORKERS = "4"; $env:SMOOTH = "0.1"; $env:RESUME = "1"
$env:ARCH = "64,128,256;4,4,3"; $env:SEED = "$Seed"; $env:TAG = $Tag; $env:T_KD = "1"; $env:A_KD = "1"; $env:BLUR = $Blur; $env:ZOOM = $Zoom; $env:BLUR_S = $BlurS; $env:ZOOM_S = $ZoomS; $env:STRIDES = $Strides; $env:GRAY = $Gray
if ($NoTeachers) { Remove-Item Env:TEACHERS -ErrorAction SilentlyContinue }  # plain cross-entropy + label smoothing, no distillation
elseif ($Teachers) { $env:TEACHERS = $Teachers } else { $env:TEACHERS = (@("best_t2_400ep.pt", "best_t2_400ep_s14.pt", "best_t2_400ep_s15.pt", "best_t2_600ep.pt") | ForEach-Object { Join-Path $Ck $_ }) -join "," }
"$(Get-Date -Format HH:mm) start blur student $Tag ($Epochs ep, seed $Seed, BLUR=$Blur/$BlurS, ZOOM=$Zoom/$ZoomS, strides $Strides, full $Full, gray $Gray, cutmix $Mix, erase $Erase, zoomin $ZoomIn, teachers '$env:TEACHERS')" | Add-Content pipeline.log
py -3.11 -u train.py *>> "kd$Tag.log"
py -3.11 tta_balance.py "best$Tag.pt" "sub$Tag" *>> "kd$Tag.log"
"$(Get-Date -Format HH:mm) done $Tag -> sub${Tag}_tta.csv" | Add-Content pipeline.log
