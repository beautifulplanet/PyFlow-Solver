@echo off
REM Run script for the PyFlow hybrid solver benchmark

echo ==================================
echo Running PyFlow Benchmark
echo ==================================

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found in PATH
    exit /b 1
)

REM Run the benchmark script
echo Starting benchmark...
python benchmark_hybrid.py
if %ERRORLEVEL% NEQ 0 (
    echo Error: Benchmark failed
    exit /b 1
)

echo ==================================
echo Benchmark completed!
echo ==================================
echo Results are available in benchmark_results.csv and PNG image files.
