# PowerShell script: precise, timestamped terminal output artifact
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "C:\Users\Elite\Documents\commands\terminal_logs"
if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$terminalLog = "$logDir\terminal_output_$timestamp.log"
Start-Transcript -Path $terminalLog -Append
Write-Host "Terminal logging started. Output will be saved to $terminalLog."
# To stop logging, run: Stop-Transcript
