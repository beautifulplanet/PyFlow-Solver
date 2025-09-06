# PowerShell script to extract error punchlist from pytest log
$pytestLog = "C:\Users\Elite\Documents\commands\pytest_errors.log"
$punchlist = "C:\Users\Elite\Documents\commands\pytest_punchlist.txt"
Select-String -Path $pytestLog -Pattern "FAILED|ERROR|AssertionError" | Out-File $punchlist
Write-Host "Punchlist created at $punchlist."
