# PowerShell script to run pytest and save all errors to a log file
$pytestLog = "C:\Users\Elite\Documents\commands\pytest_errors.log"
pytest > $pytestLog 2>&1
Write-Host "Pytest run complete. Errors and output saved to $pytestLog."
