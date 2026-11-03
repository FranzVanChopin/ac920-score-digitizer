param([string]$VitisRoot='D:\Xlinix\Vitis\2024.2')
$ErrorActionPreference='Stop'
$sourceRoot=Split-Path $PSScriptRoot -Parent
$dpRoot=Split-Path (Split-Path $sourceRoot -Parent) -Parent
$testBuild=Join-Path $dpRoot 'build'
New-Item -ItemType Directory -Path $testBuild -Force | Out-Null
$compiler=Join-Path $VitisRoot 'tps/mingw/10.0.0/win64.o/nt/bin/gcc.exe'
foreach($component in @('capture_onecard','capture_panel')) {
    $binary=Join-Path $testBuild "test_$component.exe"
    $extra=@()
    if($component -eq 'capture_panel') {$extra=@((Join-Path $sourceRoot 'capture_button.c'))}
    & $compiler -std=c11 -Wall -Wextra -Werror -pedantic -O2 -I $sourceRoot `
        (Join-Path $sourceRoot "$component.c") (Join-Path $PSScriptRoot "test_$component.c") @extra -o $binary
    if($LASTEXITCODE) {throw "Onecard test build failed: $component"}
    & $binary
    if($LASTEXITCODE) {throw "Onecard test failed: $component"}
}
& (Join-Path $PSScriptRoot 'run_dp_tests.ps1') -VitisRoot $VitisRoot
