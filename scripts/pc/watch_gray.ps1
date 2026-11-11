# Live view of the grayscale test (run_gray.ps1). Refreshes every 10 s; Ctrl-C to stop (training keeps running).
Set-Location (Join-Path $PSScriptRoot "..\..")
while ($true) {
    Clear-Host
    "Grayscale test: baseline _bz45 vs _gray45 (45k train, 5k held-out val)   $(Get-Date -Format HH:mm:ss)"
    foreach ($tag in "_bz45", "_gray45") {
        $f = "kd$tag.log"
        ""; "=== $tag ==="
        if (-not (Test-Path $f)) { "  not started yet"; continue }
        $parts = (Get-Content $f -Raw) -split "[`r`n]+" | Where-Object { $_.Trim() }
        $parts | Where-Object { $_ -match "^ep\s+\d+/\d+\s+loss .* val acc" } | Select-Object -Last 8 | ForEach-Object { "  " + $_.TrimEnd() }
        $last = $parts | Select-Object -Last 1
        if ($last -match "\[") { "  now: " + $last.TrimEnd() }
    }
    Start-Sleep 10
}
