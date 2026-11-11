# SupCon test (PC): 200 contrastive epochs + linear head. Usage: -Epochs N
param([int]$Epochs = 200)
$env:EPOCHS="$Epochs"; $env:TAG="_supcon"
python -u supcon.py *> supcon.log
