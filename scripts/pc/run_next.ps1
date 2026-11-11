# Runs 28 and 29 (PC), one after the other, each = the 0.911 recipe (BLUR=0.2/1.0, ZOOM=0.3/0.6) with ONE change:
#  _delay : delayed downsampling, STRIDES=1,2,1 (stage 3 at 16x16 instead of 8x8; same 4,697,162 params, ~1.4x student compute)
#  _bteach: better teachers = our strongest students (0.911, 0.902, 0.898 on Kaggle) + the 600-epoch teacher, instead of the 4 original teachers
# Outputs sub_delay_tta.csv then sub_bteach_tta.csv. Resumable: re-run this script.
$here = $PSScriptRoot; $root = Join-Path $here "..\.."; $Ck = "checkpoints"
& (Join-Path $here "run_blur.ps1") -Tag "_delay" -Strides "1,2,1"
$t = @((Join-Path $root "best_blurzoom.pt"), (Join-Path $Ck "best_kd4.pt"), (Join-Path $root "best_blurzoom2.pt"), (Join-Path $Ck "best_t2_600ep.pt")) -join ","
& (Join-Path $here "run_blur.ps1") -Tag "_bteach" -Teachers $t
