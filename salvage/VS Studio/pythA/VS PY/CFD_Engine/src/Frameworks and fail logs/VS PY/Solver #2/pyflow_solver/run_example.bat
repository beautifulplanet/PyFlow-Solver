@echo off
REM Run script for the PyFlow hybrid solver example

echo ==================================
echo Running PyFlow Hybrid Solver Example
echo ==================================

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found in PATH
    exit /b 1
)

REM Run the example script
python hybrid_example.py
if %ERRORLEVEL% NEQ 0 (
    echo Error: Example script failed
    exit /b 1
)

echo ==================================
echo Example completed!
echo ==================================
echo Results are saved as PNG files in the current directory.
