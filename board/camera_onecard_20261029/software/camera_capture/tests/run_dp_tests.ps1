param([string]$VitisRoot='D:\Xlinix\Vitis\2024.2')
$ErrorActionPreference='Stop'
$sourceRoot=Split-Path $PSScriptRoot -Parent
$dpRoot=Split-Path (Split-Path $sourceRoot -Parent) -Parent
$testBuild=Join-Path $dpRoot 'build'
New-Item -ItemType Directory -Path $testBuild -Force | Out-Null
$compiler=Join-Path $VitisRoot 'tps/mingw/10.0.0/win64.o/nt/bin/gcc.exe'
foreach($component in @('dp_pixels','capture_dp')) {
    $binary=Join-Path $testBuild "test_$component.exe"
    $defines=@()
    if($component -eq 'capture_dp') {$defines=@('-DCAPTURE_DP_HOST_TEST')}
    & $compiler -std=c11 -Wall -Wextra -Werror -pedantic -O2 @defines `
        -I $sourceRoot -I $PSScriptRoot (Join-Path $sourceRoot 'dp_pixels.c') `
        (Join-Path $PSScriptRoot "test_$component.c") -o $binary
    if($LASTEXITCODE) {throw "DP test build failed: $component"}
    & $binary
    if($LASTEXITCODE) {throw "DP test failed: $component"}
}
