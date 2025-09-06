# run_all_tests.ps1
# PowerShell script to build and run all Fortran unit tests in the tests/ directory
# Usage: .\run_all_tests.ps1

$ErrorActionPreference = 'Stop'

$gccPath = 'C:\msys64\ucrt64\bin'
if (-not ($env:Path -split ';' | Where-Object { $_ -eq $gccPath })) {
    $env:Path += ";$gccPath"
}

$build = 'build'
$tests = 'tests'
$fflags = @('-O2','-Wall','-Wextra','-fimplicit-none','-std=f2008')

if (-not (Test-Path $build)) { New-Item -ItemType Directory -Path $build | Out-Null }

$core = @('parameters.f90','fields.f90','boundary_conditions.f90','io_utils.f90','solver.f90')
$coreObjs = @()
foreach ($f in $core) {
    $obj = Join-Path $build ($f -replace '.f90','.o')
    if (-not (Test-Path $obj) -or (Get-Item (Join-Path 'src' $f)).LastWriteTime -gt (Get-Item $obj).LastWriteTime) {
        Write-Host "Compiling $f" -ForegroundColor Cyan
        & gfortran @fflags -c (Join-Path 'src' $f) -o $obj
    }
    $coreObjs += $obj
}

$testFiles = Get-ChildItem $tests -Filter 'test_*.f90' | Select-Object -ExpandProperty Name
$results = @()
foreach ($test in $testFiles) {
    $testName = [System.IO.Path]::GetFileNameWithoutExtension($test)
    $obj = Join-Path $build ($test -replace '.f90','.o')
    Write-Host "Compiling $test" -ForegroundColor Yellow
    & gfortran @fflags -c (Join-Path $tests $test) -o $obj
    $exe = Join-Path $build ($testName + '.exe')
    Write-Host "Linking $exe" -ForegroundColor Yellow
    & gfortran @fflags -o $exe ($coreObjs + $obj)
    Write-Host "Running $exe..." -ForegroundColor Green
    $output = & $exe 2>&1
    $results += "==== $testName ===="
    $results += $output
    $results += ""
}

Write-Host "\nTest Results Summary:" -ForegroundColor Magenta
$results | ForEach-Object { Write-Host $_ }
