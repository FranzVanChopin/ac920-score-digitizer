$ErrorActionPreference='Stop'
$labRoot=Split-Path $PSScriptRoot -Parent
$sim=Join-Path $labRoot 'build/keys_sim'
New-Item -ItemType Directory -Force -Path $sim | Out-Null
$sdk='D:/Xlinix/Vivado/2024.2/bin'
Push-Location $sim
try {
    & "$sdk/xvlog.bat" -sv "$labRoot/rtl/camera/camera_key_sync.v" "$labRoot/rtl/camera/tb_camera_key_sync.sv"
    if($LASTEXITCODE) {throw 'key RTL compile failed'}
    & "$sdk/xelab.bat" tb_camera_key_sync -s keys
    if($LASTEXITCODE) {throw 'key RTL elaboration failed'}
    & "$sdk/xsim.bat" keys -runall
    if($LASTEXITCODE) {throw 'key RTL simulation failed'}
} finally {Pop-Location}
