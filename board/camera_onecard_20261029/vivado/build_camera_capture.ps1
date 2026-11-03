param(
    [string]$Vivado = 'D:\Xlinix\Vivado\2024.2\bin\vivado.bat'
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if (-not (Test-Path -LiteralPath $Vivado)) { throw "Vivado not found: $Vivado" }
$buildDir = Join-Path $repoRoot 'build\vivado_camera'
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
$drive = $null
$mappingCreated = $false
$oldGenerate = $env:CAMERA_GENERATE
try {
    # HLS-generated filenames exceed Windows MAX_PATH under the normal workspace.
    # Only create a new mapping; never reuse or remove someone else's drive.
    foreach ($letter in 'R','S','T','U','V','W','X','Y','Z') {
        if (-not (Get-PSDrive -Name $letter -ErrorAction SilentlyContinue) -and
            -not (Test-Path -LiteralPath "${letter}:\")) {
            $drive = "${letter}:"
            break
        }
    }
    if ($null -eq $drive) { throw 'No free drive letter R:..Z: for the Vivado short-path workaround.' }
    & subst.exe $drive $repoRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the temporary workspace drive.' }
    $mappingCreated = $true
    $shortRoot = "$drive/"
    $env:CAMERA_GENERATE = '1'
    & $Vivado -mode batch -nojournal -log "${shortRoot}build/vivado_camera/create_camera.log" -source "${shortRoot}vivado/create_camera_capture.tcl"
    if ($LASTEXITCODE -ne 0) { throw 'Camera block-design generation failed.' }
    & $Vivado -mode batch -nojournal -log "${shortRoot}build/vivado_camera/build_camera.log" -source "${shortRoot}vivado/build_camera_capture.tcl"
    if ($LASTEXITCODE -ne 0) { throw 'Camera hardware build failed; inspect build/vivado_camera logs.' }
    Write-Host "Camera bitstream and XSA ready in $buildDir"
}
finally {
    $env:CAMERA_GENERATE = $oldGenerate
    if ($mappingCreated) {
        $ownMapping = (& subst.exe) | Where-Object {
            $_ -match ('^' + [regex]::Escape($drive) + '\\: => ' + [regex]::Escape($repoRoot) + '$')
        }
        if ($ownMapping) { & subst.exe $drive /D }
    }
}
