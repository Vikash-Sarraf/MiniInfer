param(
    [Parameter(Mandatory = $true)]
    [string]$Command,

    [int]$SampleMs = 100
)

$ErrorActionPreference = "Stop"

$shellPath = (Get-Process -Id $PID).Path
if ([string]::IsNullOrWhiteSpace($shellPath)) {
    $shellPath = "powershell"
}

$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Command))

$process = Start-Process $shellPath `
    -ArgumentList "-NoProfile", "-EncodedCommand", $encodedCommand `
    -PassThru `
    -NoNewWindow

Write-Host "Started process tree root PID: $($process.Id)"
Write-Host "Command: $Command"
Write-Host "Sampling every ${SampleMs}ms..."

$startTime = Get-Date
$lastSampleTime = $startTime
$lastCpuSeconds = 0.0
$totalCpuSeconds = 0.0
$peakCpuPercent = 0.0
$logicalProcessorCount = [Environment]::ProcessorCount
$peakWorkingSet = [int64]0
$peakPrivateMemory = [int64]0
$peakPagedMemory = [int64]0

function Get-DescendantProcessIds($ParentProcessId) {
    $children = @(
        Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentProcessId" -ErrorAction SilentlyContinue
    )

    foreach ($child in $children) {
        if ($null -ne $child.ProcessId) {
            $childId = [int]$child.ProcessId
            $childId
            Get-DescendantProcessIds $childId
        }
    }
}

function Sample-ProcessTree($RootProcessId) {
    $ids = @([int]$RootProcessId) + @(Get-DescendantProcessIds $RootProcessId)
    $ids = @($ids | Where-Object { $null -ne $_ } | Select-Object -Unique)

    $workingSet = [int64]0
    $privateMemory = [int64]0
    $pagedMemory = [int64]0
    $cpuSeconds = 0.0

    foreach ($id in $ids) {
        $sample = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($null -ne $sample) {
            $workingSet += $sample.WorkingSet64
            $privateMemory += $sample.PrivateMemorySize64
            $pagedMemory += $sample.PagedMemorySize64
            if ($null -ne $sample.CPU) {
                $cpuSeconds += $sample.CPU
            }
        }
    }

    $now = Get-Date
    $elapsedSinceLastSample = ($now - $script:lastSampleTime).TotalSeconds
    $cpuDelta = $cpuSeconds - $script:lastCpuSeconds

    if ($elapsedSinceLastSample -gt 0.0 -and $cpuDelta -ge 0.0 -and $script:logicalProcessorCount -gt 0) {
        $cpuPercent = ($cpuDelta / $elapsedSinceLastSample / $script:logicalProcessorCount) * 100.0
        $script:peakCpuPercent = [Math]::Max($script:peakCpuPercent, $cpuPercent)
    }

    $script:lastSampleTime = $now
    $script:lastCpuSeconds = $cpuSeconds
    $script:totalCpuSeconds = [Math]::Max($script:totalCpuSeconds, $cpuSeconds)
    $script:peakWorkingSet = [Math]::Max($script:peakWorkingSet, $workingSet)
    $script:peakPrivateMemory = [Math]::Max($script:peakPrivateMemory, $privateMemory)
    $script:peakPagedMemory = [Math]::Max($script:peakPagedMemory, $pagedMemory)
}

Sample-ProcessTree $process.Id

while (-not $process.HasExited) {
    Sample-ProcessTree $process.Id
    Wait-Event -Timeout ($SampleMs / 1000.0) | Out-Null
    $process.Refresh()
}

$process.WaitForExit()
$process.Refresh()
Sample-ProcessTree $process.Id

function Format-Bytes($bytes) {
    if ($bytes -ge 1GB) {
        return "{0:N3} GiB" -f ($bytes / 1GB)
    }
    if ($bytes -ge 1MB) {
        return "{0:N3} MiB" -f ($bytes / 1MB)
    }
    if ($bytes -ge 1KB) {
        return "{0:N3} KiB" -f ($bytes / 1KB)
    }
    return "$bytes bytes"
}

$wallSeconds = ($process.ExitTime - $process.StartTime).TotalSeconds
if ($wallSeconds -le 0.0) {
    $wallSeconds = ((Get-Date) - $startTime).TotalSeconds
}
$averageCpuPercent = 0.0
if ($wallSeconds -gt 0.0 -and $logicalProcessorCount -gt 0) {
    $averageCpuPercent = ($totalCpuSeconds / $wallSeconds / $logicalProcessorCount) * 100.0
}

Write-Host "Exit code: $($process.ExitCode)"
Write-Host ("Wall time:                {0:N3}s" -f $wallSeconds)
Write-Host ("Total tree CPU time:      {0:N3}s" -f $totalCpuSeconds)
Write-Host "Logical processors:       $logicalProcessorCount"
Write-Host ("Average tree CPU:         {0:N1}%" -f $averageCpuPercent)
Write-Host ("Peak sampled tree CPU:    {0:N1}%" -f $peakCpuPercent)
Write-Host "Peak tree working set:    $(Format-Bytes $peakWorkingSet)"
Write-Host "Peak tree private memory: $(Format-Bytes $peakPrivateMemory)"
Write-Host "Peak tree paged memory:   $(Format-Bytes $peakPagedMemory)"