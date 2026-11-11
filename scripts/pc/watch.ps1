# Live view of a detached run's log: last epochs, the current progress bar, and an ETA. Ctrl-C to stop (the run keeps going).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\pc\watch.ps1 [-Log teacher_big45.log]
param([string]$Log = "teacher_big45.log")
Set-Location (Join-Path $PSScriptRoot "..\..")
$t0 = $null; $e0 = 0
while ($true) {
    try {  # the run holds the file open, so read with shared access; PowerShell *> logs are UTF-16
        $fs = [IO.File]::Open((Resolve-Path $Log), 'Open', 'Read', 'ReadWrite')
        $text = (New-Object IO.StreamReader($fs, [Text.Encoding]::Unicode)).ReadToEnd(); $fs.Close()
    } catch { Clear-Host; "waiting for $Log ..."; Start-Sleep 5; continue }
    $parts = $text -split "[`r`n]" | Where-Object { $_.Trim() }
    $epochs = @($parts | Where-Object { $_ -match 'val acc' })
    $bar = $parts | Select-Object -Last 1
    Clear-Host
    "$Log   $(Get-Date -Format HH:mm:ss)"
    ($parts | Select-Object -First 1)
    ""
    $epochs | Select-Object -Last 8 | ForEach-Object { $_.TrimEnd() }
    ""
    $bar.TrimEnd()
    if ($epochs.Count -and $epochs[-1] -match 'ep\s+(\d+)/(\d+)') {
        $done, $total = [int]$Matches[1], [int]$Matches[2]
        if (-not $t0) { $t0 = Get-Date; $e0 = $done }
        elseif ($done -gt $e0) {
            $per = ((Get-Date) - $t0).TotalSeconds / ($done - $e0)
            "~{0:n0} s/epoch, finishes about {1:HH:mm}" -f $per, (Get-Date).AddSeconds($per * ($total - $done))
        }
    }
    if ($text -match 'wrote submission|Traceback') { "run finished"; break }
    Start-Sleep 3
}
