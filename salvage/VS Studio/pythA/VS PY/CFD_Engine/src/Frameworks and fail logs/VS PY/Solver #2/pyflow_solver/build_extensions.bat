@echo off
REM Build script for compiling the PyFlow C++ extensions

echo ==================================
echo Building PyFlow C++ Extensions
echo ==================================

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found in PATH
    exit /b 1
)

REM Check for required packages
echo Checking dependencies...
python -c "import pybind11" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing pybind11...
    python -m pip install pybind11
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install pybind11. Please install manually with:
        echo pip install pybind11
        exit /b 1
    )
)

python -c "import numpy" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing numpy...
    python -m pip install numpy
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install numpy. Please install manually with:
        echo pip install numpy
        exit /b 1
    )
)

python -c "import matplotlib" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing matplotlib...
    python -m pip install matplotlib
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install matplotlib. Please install manually with:
        echo pip install matplotlib
        exit /b 1
    )
)

REM Build the extensions
echo Building C++ extensions...
python setup.py build_ext --inplace
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to build C++ extensions
    exit /b 1
)

echo ==================================
echo Successfully built PyFlow C++ extensions!
echo ==================================
