param(
    [Parameter(Mandatory=$true)][string]$VivadoRoot,
    [Parameter(Mandatory=$true)][string]$VitisRoot,
    [string]$Python='python',
    [switch]$SkipHardware
)
$ErrorActionPreference='Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$base = Join-Path $repo 'board/camera_onecard_20261029'
$current = Join-Path $repo 'board/camera_interactive_20261030'
$env:VITIS_ROOT = $VitisRoot
$env:CAMERA_WORKSPACE = $repo
if (-not $SkipHardware) {
    & (Join-Path $base 'vivado/build_camera_capture.ps1') -Vivado (Join-Path $VivadoRoot 'bin/vivado.bat')
    if ($LASTEXITCODE) { throw 'Vivado build failed' }
}
$xsa = Join-Path $base 'build/vivado_camera/camera_capture.xsa'
if (!(Test-Path -LiteralPath (Join-Path $base 'build/vivado_camera/hardware.complete'))) { throw 'Validated pure DP hardware required' }
& (Join-Path $VitisRoot 'bin/xsct.bat') (Join-Path $base 'software/camera_capture/build_vitis.tcl') bsp $xsa
if ($LASTEXITCODE) { throw 'BSP/FSBL/PMU build failed' }
& $Python (Join-Path $current 'build_onecard.py')
if ($LASTEXITCODE) { throw 'Current Light2 firmware build failed' }
Write-Host (Join-Path $current 'build/BOOT.BIN')
