# t2 recipe, 600 epochs. Output: best_t2_600ep.pt, submission_t2_600ep.csv, log t2_600ep.log
# Don't edit train.py while this runs: Windows data-loader workers re-read it from disk.
Set-Location $PSScriptRoot
Remove-Item Env:TEACHERS, Env:FOLD, Env:SMOOTH, Env:T_KD -ErrorAction SilentlyContinue
$env:EPOCHS = "600"; $env:FULL = "1"; $env:MIX = "1"; $env:WORKERS = "4"
$env:ARCH = "64,128,256;4,4,3"; $env:SEED = "12"; $env:TAG = "_t2_600ep"
python -u train.py *> t2_600ep.log
