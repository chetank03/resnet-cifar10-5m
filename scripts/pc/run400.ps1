# t2 recipe, 400 epochs. Output: best_t2_400ep.pt, submission_t2_400ep.csv, log t2_400ep.log
Set-Location $PSScriptRoot
Remove-Item Env:TEACHERS -ErrorAction SilentlyContinue
$env:EPOCHS = "400"; $env:FULL = "1"; $env:MIX = "1"
$env:ARCH = "64,128,256;4,4,3"; $env:SEED = "11"; $env:TAG = "_t2_400ep"
python -u train.py *> t2_400ep.log
