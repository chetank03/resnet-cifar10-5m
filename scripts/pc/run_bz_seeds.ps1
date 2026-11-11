# Reseeds of the 0.911 recipe (run_blur.ps1 defaults: BLUR=0.2, ZOOM=0.3, CutMix, 4 teachers, 50k, 400 ep).
# Waits for any running run_blur.ps1 (e.g. _zoomin) to finish first, since there is one GPU.
# Output: sub_bz<seed>_tta.csv per seed. Differences between seeds are noise (+-0.9 pts on 1,000 public images).
param([int[]]$Seeds = @(42, 43))
while (Get-CimInstance Win32_Process -Filter "name='powershell.exe'" | Where-Object { $_.CommandLine -like "*run_blur.ps1*" }) { Start-Sleep 60 }
foreach ($s in $Seeds) { & (Join-Path $PSScriptRoot "run_blur.ps1") -Seed $s -Tag "_bz$s" }
