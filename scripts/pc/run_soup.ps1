# Run 19 (PC): model soup. Fine-tune the base student N times (different seeds, same KD recipe, LR 0.02, short cosine),
# then average the weights with soup.py: still ONE 4.70M net, so it is a valid entry. Soups mainly help under
# distribution shift (Wortsman et al. 2022), which is where our test images sit (val 0.972 vs public 0.90).
# Honest check: -Init best_kd45.pt -Full 0 (soup.py prints held-out val for each piece and the soup).
# Final:        -Init <Desktop>\best_kd4.pt -Full 1 -Tag _soup50, then tta_balance.py best_soup50.pt subsoup50.
param([string]$Init = "best_kd45.pt", [int]$Full = 0, [int]$N = 4, [int]$Epochs = 60, [string]$LR = "0.02", [string]$Tag = "_soup45",
      [string]$Ck = "checkpoints")
Set-Location (Join-Path $PSScriptRoot "..\..")
if (-not (Test-Path $Init)) { throw "base checkpoint $Init not found" }
Remove-Item Env:FOLD, Env:SAM, Env:TEACHER -ErrorAction SilentlyContinue
$env:EPOCHS = "$Epochs"; $env:FULL = "$Full"; $env:MIX = "1"; $env:WORKERS = "4"; $env:SMOOTH = "0.1"; $env:LR = $LR
$env:ARCH = "64,128,256;4,4,3"; $env:T_KD = "1"; $env:A_KD = "1"; $env:INIT = $Init
$env:TEACHERS = (@("best_t2_400ep.pt", "best_t2_400ep_s14.pt", "best_t2_400ep_s15.pt", "best_t2_600ep.pt") | ForEach-Object { Join-Path $Ck $_ }) -join ","
$parts = @()
for ($i = 1; $i -le $N; $i++) {
    $env:SEED = "$(100 + $i)"; $env:TAG = "${Tag}_$i"
    py -3.11 -u train.py *> "kd$env:TAG.log"
    $parts += "best$env:TAG.pt"
}
py -3.11 soup.py "best$Tag.pt" @parts *> "soup$Tag.log"
if ($Full -eq 1) { py -3.11 tta_balance.py "best$Tag.pt" "sub$Tag" *>> "soup$Tag.log" }
