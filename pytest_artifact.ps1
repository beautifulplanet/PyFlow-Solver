# PowerShell script: precise, timestamped pytest log and punchlist artifact
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "C:\Users\Elite\Documents\commands\pytest_logs"
if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$pytestLog = "$logDir\pytest_errors_$timestamp.log"
$punchlist = "$logDir\pytest_punchlist_$timestamp.txt"

# Run pytest and save all output with timestamp
pytest > $pytestLog 2>&1

# Extract errors and failures for punchlist
Select-String -Path $pytestLog -Pattern "FAILED|ERROR|AssertionError" | Out-File $punchlist

Write-Host "Pytest run complete. Logs and punchlist saved as artifacts:"
Write-Host $pytestLog
Write-Host $punchlist
