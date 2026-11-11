# Run 16 (PC): big distillation teacher, WRN-28-10-shaped (160,320,640)x(4,4,4), ~36M params (TEACHER=1 lifts the 5M cap; never submitted).
# Rules allow any teacher trained from scratch on the course data. Same recipe as t2 (CutMix, TrivialAugment, EMA, smoothing 0.1).
# Default -Full 0 holds out 5k so the val acc is honest: the plan only continues if it clears ~0.978 (vs 0.972 for the 4.7M nets).
# Usage: -Full 1 -Tag _big50 for the 50k teacher used by the final student.  Output: best{Tag}.pt, log teacher{Tag}.log.
param([int]$Epochs = 200, [int]$Full = 0, [int]$Seed = 21, [string]$Tag = "_big45")
Set-Location (Join-Path $PSScriptRoot "..\..")
Remove-Item Env:FOLD, Env:SAM, Env:TEACHERS -ErrorAction SilentlyContinue
$env:TEACHER = "1"; $env:ARCH = "160,320,640;4,4,4"
$env:EPOCHS = "$Epochs"; $env:FULL = "$Full"; $env:MIX = "1"; $env:WORKERS = "4"; $env:SMOOTH = "0.1"
$env:SEED = "$Seed"; $env:TAG = $Tag
py -3.11 -u train.py *> "teacher$Tag.log"  # 3.11 has the CUDA torch build; the default `python` is now 3.14 without it
