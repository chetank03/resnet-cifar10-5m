# Run 11: SE + stochastic depth teacher, t2 recipe, 400 epochs (PC). Usage: -Seed N
param([int]$Seed = 16)
$env:EPOCHS="400"; $env:FULL="1"; $env:MIX="1"; $env:ARCH="64,128,256;4,4,3"; $env:WORKERS="4"; $env:SMOOTH="0.1"
$env:SE="1"; $env:DROP_PATH="0.1"; $env:SEED="$Seed"; $env:TAG="_se_s$Seed"
python -u train.py *> "se_s$Seed.log"
python predict.py "best_se_s$Seed.pt" "submission_se_s$Seed.csv" *>> "se_s$Seed.log"
