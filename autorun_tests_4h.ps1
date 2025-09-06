# PowerShell script to autorun all tests and artifact/error logging for 4 hours
# Usage: powershell -ExecutionPolicy Bypass -File autorun_tests_4h.ps1

$startTime = Get-Date
$maxHours = 4
$logDir = "pytest_logs"
if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

Write-Host "Starting autorun test loop for $maxHours hours..."

while ($true) {
    $now = Get-Date
    $elapsed = ($now - $startTime).TotalHours
    if ($elapsed -ge $maxHours) {
        Write-Host "Reached $maxHours hours. Exiting."
        break
    }
    $timestamp = $now.ToString("yyyyMMdd_HHmmss")
    $logFile = Join-Path $logDir "autorun_pytest_$timestamp.log"
    $errFile = Join-Path $logDir "autorun_pytest_errors_$timestamp.log"
    $summaryFile = Join-Path $logDir "autorun_pytest_summary_$timestamp.txt"

    Write-Host "[$timestamp] Running tests..."
    try {
        pytest > $logFile 2>&1
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            Write-Host "[$timestamp] Test run failed. Logging errors."
            Select-String -Path $logFile -Pattern "FAIL|ERROR|Traceback|AssertionError" | Out-File $errFile
        }
        # Summarize failures
        Select-String -Path $logFile -Pattern "FAILED|ERROR|Traceback|AssertionError" | Out-File $summaryFile
    } catch {
        Write-Host "[$timestamp] Exception during test run: $_"
        Add-Content $errFile "Exception: $_"
    }
    Start-Sleep -Seconds 10  # Short pause between runs
}

Write-Host "Autorun test loop complete."
