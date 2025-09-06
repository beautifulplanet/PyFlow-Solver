# PowerShell script to log all terminal output in this session
$logPath = "C:\Users\Elite\Documents\commands\terminal_output.log"
Start-Transcript -Path $logPath -Append
Write-Host "Logging started. All terminal output will be saved to $logPath."
# To stop logging, run: Stop-Transcript
