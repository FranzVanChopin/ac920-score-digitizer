param(
    [string]$VitisRoot = $(if ($env:VITIS_ROOT) { $env:VITIS_ROOT } else { 'D:\Xlinix\Vitis\2024.2' }),
    [string]$Compiler = ''
)
$ErrorActionPreference = "Stop"
if (!$Compiler) { $Compiler = Join-Path $VitisRoot 'tps\mingw\10.0.0\win64.o\nt\bin\gcc.exe' }
$captureRoot = Split-Path $PSScriptRoot -Parent
$workspaceRoot = Split-Path (Split-Path $captureRoot -Parent) -Parent
$testBuild = Join-Path $workspaceRoot "build\camera_capture_tests"
New-Item -ItemType Directory -Force -Path $testBuild | Out-Null
if (!(Test-Path -LiteralPath $Compiler)) { throw "Host C compiler not found: $Compiler" }
$env:PATH = (Split-Path $Compiler -Parent) + ";" + $env:PATH
foreach ($component in @("capture_button", "capture_store", "capture_pair", "preview", "imx477", "capture_color", "capture_exposure", "capture_settle", "capture_binary", "capture_focus", "capture_denoise", "capture_fatfs", "capture_still")) {
    $binary = Join-Path $testBuild "test_$component.exe"
    $extraSources = @()
    $extraFlags = @()
    if ($component -eq 'capture_fatfs') { $extraFlags = @('-I',(Join-Path $PSScriptRoot 'fatfs_fixture')) }
    if ($component -eq 'imx477') {
        $extraSources += Join-Path $captureRoot 'capture_still.c'
        $extraSources += Join-Path $captureRoot 'capture_settle.c'
    }
    if ($component -eq 'capture_binary') {
        foreach ($dependency in @('capture_color.c','capture_focus.c','preview.c')) {
            $extraSources += Join-Path $captureRoot $dependency
        }
    }
    if ($component -eq 'capture_focus') { $extraSources += Join-Path $captureRoot 'preview.c' }
    & $Compiler -std=c11 -Wall -Wextra -Werror -pedantic -O2 -I $captureRoot @extraFlags `
        (Join-Path $captureRoot "$component.c") `
        (Join-Path $PSScriptRoot "test_$component.c") @extraSources -o $binary -lm
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed: $component" }
    & $binary
    if ($LASTEXITCODE -ne 0) { throw "Host tests failed: $component" }
}

# Exercise the PS I2C transaction code against real Xilinx register definitions,
# replacing only MMIO and generated BSP configuration with a hardware model.
$embedded = Join-Path $VitisRoot 'data\embeddedsw'
$iicInclude = Join-Path $embedded 'XilinxProcessorIPLib\drivers\iicps_v3_20\src'
$bspSource = Join-Path $embedded 'lib\bsp\standalone_v9_2\src'
$fixture = Join-Path $PSScriptRoot 'platform_i2c_fixture'
foreach ($variant in @('classic', 'sdt')) {
    $binary = Join-Path $testBuild "test_platform_i2c_$variant.exe"
    $defines = @()
    if ($variant -eq 'sdt') { $defines += '-DSDT' }
    & $Compiler -std=c11 -Wall -Wextra -Werror -O2 @defines `
        -include (Join-Path $fixture 'mmio_override.h') `
        -I $fixture -I $iicInclude -I (Join-Path $bspSource 'common') `
        -I (Join-Path $bspSource 'arm\ARMv8\64bit') `
        (Join-Path $captureRoot 'platform_i2c.c') `
        (Join-Path $PSScriptRoot 'test_platform_i2c.c') -o $binary
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed: platform_i2c $variant" }
    & $binary
    if ($LASTEXITCODE -ne 0) { throw "Host tests failed: platform_i2c $variant" }
}
Write-Host 'PASS: all camera capture host tests'
