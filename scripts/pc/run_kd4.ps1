# Run 9: distill the ensemble of the 4 strong full-data models into one 4.70M student.
# Pure KL at T=1 (A_KD=1, no hard-label term), same augmented view for teacher and student, 400 epochs, CutMix.
# Teachers: seeds 11/14/15 at 400 epochs + the 600-epoch model (all (64,128,256)x(4,4,3), trained on all 50k).
# Output: best_kd4.pt, submission_kd4.csv, log kd4.log.  Don't edit train.py while this runs.
Set-Location $PSScriptRoot
Remove-Item Env:FOLD, Env:SAM -ErrorAction SilentlyContinue
$env:EPOCHS = "400"; $env:FULL = "1"; $env:MIX = "1"; $env:WORKERS = "4"; $env:SMOOTH = "0.1"
$env:ARCH = "64,128,256;4,4,3"; $env:SEED = "41"; $env:TAG = "_kd4"
$env:T_KD = "1"; $env:A_KD = "1"
$env:TEACHERS = "best_t2_400ep.pt,best_t2_400ep_s14.pt,best_t2_400ep_s15.pt,best_t2_600ep.pt"
python -u train.py *> kd4.log
if (-not (Test-Path submission_kd4.csv)) { python predict.py best_kd4.pt submission_kd4.csv *>> kd4.log }
python tta_balance.py best_kd4.pt subkd4 *>> kd4.log
