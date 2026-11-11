# Blur+zoom teachers -> student. Teachers: cross-entropy (no KD), the 0.911 augmentations (BLUR=0.2, ZOOM=0.3, CutMix), 50k, 400 ep,
# so their soft labels on blurred/zoomed views are good. Teacher 1 is the existing _noteach (same recipe, seed 41).
# Student: the 0.911 recipe distilled from these 4 instead of the old t2 teachers. Output: sub_bzkd_tta.csv (upload this).
# Waits for run_bz_seeds.ps1 / run_blur.ps1 to finish (one GPU). Re-running skips teachers that already exist.
param([int[]]$Seeds = @(42, 43, 44))
Set-Location (Join-Path $PSScriptRoot "..\..")
while (Get-CimInstance Win32_Process -Filter "name='powershell.exe'" | Where-Object { $_.CommandLine -like "*run_bz_seeds.ps1*" -or $_.CommandLine -like "*run_blur.ps1*" }) { Start-Sleep 60 }
$teachers = @((Resolve-Path "checkpoints\best_noteach.pt").Path)
foreach ($s in $Seeds) {
    if (-not (Test-Path "best_nt$s.pt")) { & (Join-Path $PSScriptRoot "run_blur.ps1") -Seed $s -Tag "_nt$s" -NoTeachers }
    if (-not (Test-Path "best_nt$s.pt")) { "$(Get-Date -Format HH:mm) FAILED teacher _nt$s, student not started" | Add-Content pipeline.log; exit 1 }
    $teachers += (Resolve-Path "best_nt$s.pt").Path
}
& (Join-Path $PSScriptRoot "run_blur.ps1") -Seed 41 -Tag "_bzkd" -Teachers ($teachers -join ",")
