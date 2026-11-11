# Run 14 (PC): same as run 12 but the distillation loss is MSE on logits instead of KL (Kim et al. 2021). 400 epochs, then TTA.
param([int]$Epochs = 400, [string]$Tag = "_mse")
$teachers = @("best_t2_400ep.pt","best_t2_400ep_s14.pt","best_t2_400ep_s15.pt","best_t2_600ep.pt","best_kd4.pt","best_se_s16.pt") | Where-Object { Test-Path $_ }
$env:EPOCHS="$Epochs"; $env:FULL="1"; $env:MIX="1"; $env:ARCH="64,128,256;4,4,3"; $env:WORKERS="4"; $env:SMOOTH="0.1"
$env:SE="0"; $env:DROP_PATH="0"; $env:SEED="61"; $env:TAG=$Tag; $env:RESUME="1"
$env:TEACHERS=($teachers -join ","); $env:A_KD="1"; $env:KD_LOSS="mse"
python -u train.py *> "mse$Tag.log"
python tta_balance.py "best$Tag.pt" "sub$Tag" *>> "mse$Tag.log"
