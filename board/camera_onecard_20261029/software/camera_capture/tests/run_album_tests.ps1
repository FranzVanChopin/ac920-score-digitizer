param([string]$VitisRoot='D:\Xlinix\Vitis\2024.2')
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'run_host_tests.ps1') -VitisRoot $VitisRoot
if($LASTEXITCODE -ne 0) { throw 'Camera host suite failed' }
$sourceRoot=Split-Path $PSScriptRoot -Parent
$repositoryRoot=Split-Path (Split-Path $sourceRoot -Parent) -Parent
$testBuild=Join-Path $repositoryRoot 'build\camera_capture_tests'
$gcc=Join-Path $VitisRoot 'tps\mingw\10.0.0\win64.o\nt\bin\gcc.exe'
$fixture=Join-Path $PSScriptRoot 'fatfs_real_fixture'
$binary=Join-Path $testBuild 'test_fatfs_contiguous.exe'
& $gcc -std=c11 -Wall -Wextra -Werror -O2 -I $fixture `
    (Join-Path $sourceRoot 'ff_contiguous.c') (Join-Path $fixture 'ffunicode.c') `
    (Join-Path $PSScriptRoot 'test_fatfs_contiguous.c') -o $binary
if($LASTEXITCODE -ne 0) { throw 'FatFs contiguous-write compilation failed' }
& $binary
if($LASTEXITCODE -ne 0) { throw 'FatFs contiguous-write tests failed' }
Write-Output 'PASS all album capture host tests'
