# PowerShell script to run pytest, update logs, and punchlist automatically
$pytestLog = "C:\Users\Elite\Documents\commands\pytest_errors.log"
$punchlist = "C:\Users\Elite\Documents\commands\pytest_punchlist.txt"

# Run pytest and save all output
pytest > $pytestLog 2>&1

# Extract errors and failures for punchlist
Select-String -Path $pytestLog -Pattern "FAILED|ERROR|AssertionError" | Out-File $punchlist

Write-Host "Pytest run complete. Logs and punchlist updated."
