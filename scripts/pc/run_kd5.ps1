# Run 10: born-again round 2. Same as run_kd4.ps1 but with the run-9 student added as a 5th teacher, new seed.
# Output: best_kd5.pt, submission_kd5.csv, subkd5_tta.csv, log kd5.log.  Don't edit train.py while this runs.
Set-Location $PSScriptRoot
Remove-Item Env:FOLD, Env:SAM -ErrorAction SilentlyContinue
$env:EPOCHS = "400"; $env:FULL = "1"; $env:MIX = "1"; $env:WORKERS = "4"; $env:SMOOTH = "0.1"
$env:ARCH = "64,128,256;4,4,3"; $env:SEED = "42"; $env:TAG = "_kd5"
$env:T_KD = "1"; $env:A_KD = "1"
$env:TEACHERS = "best_t2_400ep.pt,best_t2_400ep_s14.pt,best_t2_400ep_s15.pt,best_t2_600ep.pt,best_kd4.pt"
python -u train.py *> kd5.log
if (-not (Test-Path submission_kd5.csv)) { python predict.py best_kd5.pt submission_kd5.csv *>> kd5.log }
python tta_balance.py best_kd5.pt subkd5 *>> kd5.log
