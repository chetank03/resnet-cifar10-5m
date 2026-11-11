# Run 12: final student distilled from every full-data teacher present (runs 6, 6b, 6c, 10, 11), pure KL at T=1, 400 epochs, then 10-view TTA.
$teachers = @("best_t2_400ep.pt","best_t2_400ep_s14.pt","best_t2_400ep_s15.pt","best_t2_600ep.pt","best_kd4.pt","best_se_s16.pt","best_se_s17.pt") | Where-Object { Test-Path $_ }
Write-Host "teachers: $($teachers -join ', ')"
$env:EPOCHS="400"; $env:FULL="1"; $env:MIX="1"; $env:ARCH="64,128,256;4,4,3"; $env:WORKERS="4"; $env:SMOOTH="0.1"
$env:SE="0"; $env:DROP_PATH="0"; $env:SEED="43"; $env:TAG="_final"
$env:TEACHERS=($teachers -join ","); $env:T_KD="1"; $env:A_KD="1"
python -u train.py *> final.log
if (-not (Test-Path submission_final.csv)) { python predict.py best_final.pt submission_final.csv *>> final.log }
python tta_balance.py best_final.pt subfinal *>> final.log
