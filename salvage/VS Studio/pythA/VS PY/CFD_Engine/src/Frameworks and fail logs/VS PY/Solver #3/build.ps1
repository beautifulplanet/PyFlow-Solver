
# PowerShell helper to build and run without GNU make in this shell.
# Usage:  .\build.ps1 [-Rebuild] [-Run] [-BcTest]
param(
    [switch]$Rebuild,
    [switch]$Run,
    [switch]$BcTest
)

$ErrorActionPreference = 'Stop'

# Ensure gfortran in PATH (adjust if different MSYS2 root)
$gccPath = 'C:\msys64\ucrt64\bin'
if (-not ($env:Path -split ';' | Where-Object { $_ -eq $gccPath })) {
    $env:Path += ";$gccPath"
}

$src = 'src'
$tests = 'tests'
$build = 'build'
if (-not (Test-Path $build)) { New-Item -ItemType Directory -Path $build | Out-Null }

# Source file lists
$core = @('parameters.f90','fields.f90','boundary_conditions.f90','io_utils.f90','solver.f90')
$main = 'main.f90'
$bc = 'test_bc.f90'


# Use array to avoid PowerShell splitting -O2 into -O and 2
$fflags = @('-O2','-Wall','-Wextra','-fimplicit-none','-std=f2008')
Write-Host "Fortran flags: $($fflags -join ' ')" -ForegroundColor Magenta


# Compile .f90 from src or tests dir
function Compile-Obj($file, $isTest=$false) {
    $obj = Join-Path $build ($file -replace '.f90','.o')
    if ($isTest) {
        $dir = $tests
    } else {
        $dir = $src
    }
    if ($Rebuild -or -not (Test-Path $obj) -or (Get-Item (Join-Path $dir $file)).LastWriteTime -gt (Get-Item $obj).LastWriteTime) {
        Write-Host "Compiling $file" -ForegroundColor Cyan
        & gfortran @fflags -c (Join-Path $dir $file) -o $obj
    }
    return $obj
}

# Compile core
$objs = @()
foreach ($f in $core) { $objs += Compile-Obj $f }
# Main
$mainObj = Compile-Obj $main
$exe = Join-Path $build 'hpf_cfd.exe'
Write-Host 'Linking solver...' -ForegroundColor Yellow
& gfortran @fflags -o $exe ($objs + $mainObj)

if ($BcTest) {
    $bcObj = Compile-Obj $bc $true
    $bcExe = Join-Path $build 'test_bc.exe'
    Write-Host 'Linking BC test...' -ForegroundColor Yellow
    & gfortran @fflags -o $bcExe ($objs + $bcObj)
    Write-Host 'Running BC test...' -ForegroundColor Green
    & $bcExe
}

if ($Run) {
    Write-Host 'Running solver...' -ForegroundColor Green
    & $exe
}
