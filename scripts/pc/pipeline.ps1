# Teachers (all 50k, CutMix) -> distilled student. Each run ~30-35 min on the 5070 Ti.
# Progress: Get-Content pipeline.log ; per-run logs: t1.log t2.log t3.log student.log
Set-Location $PSScriptRoot
$env:FULL = "1"; $env:MIX = "1"; Remove-Item Env:EPOCHS, Env:TEACHERS -ErrorAction SilentlyContinue

$runs = @(
  @{ TAG = "_t1"; ARCH = "64,128,256;3,3,3";       SEED = 1 },
  @{ TAG = "_t2"; ARCH = "64,128,256;4,4,3";       SEED = 2 },
  @{ TAG = "_t3"; ARCH = "64,128,256,256;2,2,2,1"; SEED = 3 }
)
foreach ($r in $runs) {
  $env:TAG = $r.TAG; $env:ARCH = $r.ARCH; $env:SEED = $r.SEED
  "$(Get-Date -f HH:mm) start $($r.TAG) $($r.ARCH)" | Tee-Object pipeline.log -Append
  python -u train.py *> "t$($r.SEED).log"
  if (-not (Test-Path "best$($r.TAG).pt")) { "$(Get-Date -f HH:mm) FAILED $($r.TAG)" | Tee-Object pipeline.log -Append; exit 1 }
}

$env:TAG = "_student"; $env:ARCH = "64,128,256;4,4,3"; $env:SEED = 7
$env:TEACHERS = "best_t1.pt,best_t2.pt,best_t3.pt"
"$(Get-Date -f HH:mm) start student (distilling from 3 teachers)" | Tee-Object pipeline.log -Append
python -u train.py *> student.log
"$(Get-Date -f HH:mm) done -> submission_student.csv" | Tee-Object pipeline.log -Append
