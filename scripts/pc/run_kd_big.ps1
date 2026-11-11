# Run 17 (PC): distill the big teacher (run 16) into the 4.70M student. Same recipe as the 0.902 entry (run_kd4.ps1):
# pure KL at T=1 (A_KD=1), teacher sees the student's CutMix/TrivialAugment view, but 1000 epochs: long KD pays off only
# when the teacher is clearly stronger (Beyer et al. 2022), which is the case this time and was not in runs 10b-13.
# Default -Full 0 -Teacher best_big45.pt: 45k train / 5k honest val (the teacher never saw them either).
# Final: -Full 1 -Teacher best_big50.pt -Tag _kdbig50, then tta_balance.py. Output: best{Tag}.pt, submission{Tag}.csv, log kd{Tag}.log.
param([int]$Epochs = 1000, [int]$Full = 0, [int]$Seed = 61, [string]$Teacher = "best_big45.pt", [string]$Tag = "_kdbig45")
Set-Location (Join-Path $PSScriptRoot "..\..")
Remove-Item Env:FOLD, Env:SAM, Env:TEACHER -ErrorAction SilentlyContinue
if (-not (Test-Path $Teacher)) { throw "teacher checkpoint $Teacher not found" }
$env:EPOCHS = "$Epochs"; $env:FULL = "$Full"; $env:MIX = "1"; $env:WORKERS = "4"; $env:SMOOTH = "0.1"
$env:ARCH = "64,128,256;4,4,3"; $env:SEED = "$Seed"; $env:TAG = $Tag
$env:T_KD = "1"; $env:A_KD = "1"; $env:TEACHERS = $Teacher
py -3.11 -u train.py *> "kd$Tag.log"
py -3.11 tta_balance.py "best$Tag.pt" "sub$Tag" *>> "kd$Tag.log"
