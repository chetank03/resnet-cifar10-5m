# Grayscale test (rule-compliant selection): train the 0.911 recipe on 45k and score on the 5k held-out training split, with and without
# random grayscale (GRAY=0.1). Same seed, so the only difference is grayscale. Compare "best" val acc at the end of kd_bz45.log vs kd_gray45.log.
& "$PSScriptRoot\run_blur.ps1" -Tag "_bz45" -Full "0" -Gray "0"
& "$PSScriptRoot\run_blur.ps1" -Tag "_gray45" -Full "0" -Gray "0.1"
