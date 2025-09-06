# PowerShell script: autorun_tests_autofix_4h.ps1
# Runs tests in a loop for 4 hours, auto-fixes import/syntax errors with black/isort, logs all actions.

$startTime = Get-Date
$maxHours = 4
$logDir = "pytest_logs"
if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

# Ensure black and isort are installed
Write-Host "Ensuring black and isort are installed..."
python -m pip install --upgrade black isort | Out-Null

Write-Host "Starting autorun test+autofix loop for $maxHours hours..."

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
    pytest > $logFile 2>&1
    $exitCode = $LASTEXITCODE
    $logContent = Get-Content $logFile -Raw
    $fixAttempted = $false

    if ($exitCode -ne 0) {
        Write-Host "[$timestamp] Test run failed. Checking for fixable errors..."
        # Check for import or syntax errors
        if ($logContent -match "ImportError|ModuleNotFoundError|SyntaxError|IndentationError") {
            Write-Host "[$timestamp] Import or syntax error detected. Running black and isort..."
            python -m black . | Out-Null
            python -m isort . | Out-Null
            $fixAttempted = $true
            Add-Content $errFile "[$timestamp] Ran black and isort due to import/syntax error."
        }
        # Log errors and summary
        Select-String -Path $logFile -Pattern "FAIL|ERROR|Traceback|AssertionError" | Out-File $errFile
        Select-String -Path $logFile -Pattern "FAILED|ERROR|Traceback|AssertionError" | Out-File $summaryFile
    }
    if ($fixAttempted) {
        Write-Host "[$timestamp] Re-running tests after auto-fix..."
        pytest >> $logFile 2>&1
        $exitCode = $LASTEXITCODE
        $logContent = Get-Content $logFile -Raw
        # Log errors and summary again
        Select-String -Path $logFile -Pattern "FAIL|ERROR|Traceback|AssertionError" | Out-File $errFile -Append
        Select-String -Path $logFile -Pattern "FAILED|ERROR|Traceback|AssertionError" | Out-File $summaryFile -Append
    }
    Start-Sleep -Seconds 10  # Short pause between runs
}

Write-Host "Autorun test+autofix loop complete."
