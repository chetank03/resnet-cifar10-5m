# Same recipe as the 0.899 run (run400.ps1), different seed. Usage: run400_seed.ps1 14
# Output: best_t2_400ep_s<seed>.pt, submission_t2_400ep_s<seed>.csv, log t2_400ep_s<seed>.log
# Don't edit train.py while this runs: Windows data-loader workers re-read it from disk.
param([int]$Seed = 14)
Set-Location $PSScriptRoot
Remove-Item Env:TEACHERS, Env:FOLD, Env:SMOOTH, Env:T_KD, Env:SAM -ErrorAction SilentlyContinue
$env:EPOCHS = "400"; $env:FULL = "1"; $env:MIX = "1"; $env:WORKERS = "4"
$env:ARCH = "64,128,256;4,4,3"; $env:SEED = "$Seed"; $env:TAG = "_t2_400ep_s$Seed"
python -u train.py *> "t2_400ep_s$Seed.log"
