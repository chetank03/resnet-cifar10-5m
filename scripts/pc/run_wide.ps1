# Run 15 (PC): wider, shallower student (128,256,320)x(2,2,1), 4.44M params (width is reported to help more than depth on shifted data).
# Same distillation as run 12 (pure KL, T=1, 400 epochs, PC teachers), then 10-view TTA.
param([int]$Epochs = 400, [string]$Tag = "_wide")
$teachers = @("best_t2_400ep.pt","best_t2_400ep_s14.pt","best_t2_400ep_s15.pt","best_t2_600ep.pt","best_kd4.pt","best_se_s16.pt") | Where-Object { Test-Path $_ }
$env:EPOCHS="$Epochs"; $env:FULL="1"; $env:MIX="1"; $env:ARCH="128,256,320;2,2,1"; $env:WORKERS="4"; $env:SMOOTH="0.1"
$env:SE="0"; $env:DROP_PATH="0"; $env:SEED="71"; $env:TAG=$Tag; $env:RESUME="1"
$env:TEACHERS=($teachers -join ","); $env:T_KD="1"; $env:A_KD="1"; $env:KD_LOSS="kl"
python -u train.py *> "wide$Tag.log"
python tta_balance.py "best$Tag.pt" "sub$Tag" *>> "wide$Tag.log"
