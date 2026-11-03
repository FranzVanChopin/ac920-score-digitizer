param(
    [string]$VitisRoot = $(if ($env:VITIS_ROOT) { $env:VITIS_ROOT } else { 'D:\Xlinix\Vitis\2024.2' }),
    [switch]$HardwareOnly
)
$ErrorActionPreference = 'Stop'
# Vitis launchers can replace PSModulePath. Import the built-in utility module
# by its absolute path so SHA256 and JSON commands remain available in PS 5.1.
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
$captureRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$captureBuild = Join-Path $captureRoot 'build\camera_capture_vitis'
$artifacts = Join-Path $captureBuild 'artifacts'
$platform = Join-Path $captureBuild 'camera_capture_platform'
$a53Tools = Join-Path $VitisRoot 'gnu\aarch64\nt\aarch64-none\bin'
$env:PATH = $a53Tools + ';' + $env:PATH
$xsa = Join-Path $captureRoot 'build\vivado_camera\camera_capture.xsa'
$bitstream = Join-Path $captureRoot 'build\vivado_camera\camera_capture.runs\impl_1\camera_capture_wrapper.bit'
$bootImage = Join-Path $artifacts 'BOOT.BIN'
$hardwareMarker = Join-Path $captureRoot 'build\vivado_camera\hardware.complete'

function Invoke-CaptureTool([string]$Executable, [string[]]$Arguments) {
    $result = & $Executable @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed: $result" }
    return ($result | Out-String).Trim()
}
function Get-ElfReport([string]$Path, [string]$ReadElf) {
    $programHeaders = Invoke-CaptureTool $ReadElf @('-lW', $Path)
    $ranges = @()
    foreach ($line in $programHeaders -split '\r?\n') {
        if ($line -match '^\s*LOAD\s+0x\w+\s+0x\w+\s+(0x\w+)\s+(0x\w+)\s+(0x\w+)\s+') {
            $start = [Convert]::ToUInt64($Matches[1].Substring(2), 16)
            $memorySize = [Convert]::ToUInt64($Matches[3].Substring(2), 16)
            $ranges += [ordered]@{
                start = ('0x{0:X}' -f $start)
                endExclusive = ('0x{0:X}' -f ($start + $memorySize))
                memoryBytes = $memorySize
                fileBytes = [Convert]::ToUInt64($Matches[2].Substring(2), 16)
            }
        }
    }
    if (!$ranges.Count) { throw "No LOAD segments in $Path" }
    return [ordered]@{
        file = $Path
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
        bytes = (Get-Item -LiteralPath $Path).Length
        loadRanges = $ranges
        programHeaders = $programHeaders
        sections = Invoke-CaptureTool $ReadElf @('-SW', $Path)
    }
}

$requiredFiles = @($xsa, $bitstream, $hardwareMarker)
if (!$HardwareOnly) { $requiredFiles += $bootImage }
foreach ($required in $requiredFiles) {
    if (!(Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing artifact: $required" }
}
$xsaHash = (Get-FileHash -LiteralPath $xsa).Hash
$bitHash = (Get-FileHash -LiteralPath $bitstream).Hash
# Read a plain .NET string: Windows PowerShell 5.1's JSON serializer otherwise
# walks Get-Content's attached PSDrive/provider metadata recursively.
$markerText = [IO.File]::ReadAllText($hardwareMarker)
if ((Get-Item -LiteralPath $hardwareMarker).LastWriteTimeUtc -lt (Get-Item -LiteralPath $xsa).LastWriteTimeUtc -or
    (Get-Item -LiteralPath $hardwareMarker).LastWriteTimeUtc -lt (Get-Item -LiteralPath $bitstream).LastWriteTimeUtc) {
    throw 'Hardware was changed after its validation marker. Rerun hardware validation.'
}
if ($markerText -notmatch '(?m)^XSA SHA256:\s*([0-9a-fA-F]{64})\s*$' -or $Matches[1] -ne $xsaHash) {
    throw 'XSA does not match the validated hardware marker.'
}
if ($markerText -notmatch '(?m)^Bitstream SHA256:\s*([0-9a-fA-F]{64})\s*$' -or $Matches[1] -ne $bitHash) {
    throw 'Bitstream does not match the validated hardware marker.'
}
if ($xsaHash -ne (Get-FileHash -LiteralPath (Join-Path $platform 'hw\camera_capture.xsa')).Hash) {
    throw 'The BSP platform and delivered XSA differ. Rebuild the firmware first.'
}
$sourceHashes = @()
foreach ($source in Get-ChildItem -LiteralPath $PSScriptRoot -File | Where-Object { $_.Extension -in '.c', '.h', '.ld' }) {
    $hash = (Get-FileHash -LiteralPath $source.FullName).Hash
    $copy = Join-Path $captureBuild ('camera_capture\src\' + $source.Name)
    if ($hash -ne (Get-FileHash -LiteralPath $copy).Hash) { throw "Stale compiled source copy: $($source.Name)" }
    $sourceHashes += [ordered]@{ file = $source.Name; sha256 = $hash }
}

# A final hardware handoff must carry the exact bitstream used in BOOT.BIN.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($xsa)
try {
    $bitEntries = @($archive.Entries | Where-Object { $_.FullName -like '*.bit' })
    if ($bitEntries.Count -ne 1) { throw 'Final XSA must contain one bitstream.' }
    $stream = $bitEntries[0].Open()
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $embeddedHash = ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '') }
    finally { $stream.Dispose(); $sha.Dispose() }
    if ($embeddedHash -ne $bitHash) { throw 'XSA and external bitstreams differ.' }
} finally { $archive.Dispose() }
if ($HardwareOnly) {
    Write-Host 'PASS: current hardware validation marker, matching XSA/bitstream/BSP, and current source copies.'
    exit 0
}

$readElf = Join-Path $a53Tools 'aarch64-none-elf-readelf.exe'
$application = Join-Path $artifacts 'camera_capture.elf'
$elfReports = @(
    (Get-ElfReport $application $readElf),
    (Get-ElfReport (Join-Path $artifacts 'fsbl.elf') $readElf),
    (Get-ElfReport (Join-Path $artifacts 'pmufw.elf') (Join-Path $VitisRoot 'gnu\microblaze\nt\bin\mb-readelf.exe'))
)
$memoryRanges = @(foreach ($elf in $elfReports) {
    foreach ($range in $elf.loadRanges) {
        [PSCustomObject]@{
            start = [Convert]::ToUInt64($range.start.Substring(2), 16)
            end = [Convert]::ToUInt64($range.endExclusive.Substring(2), 16)
            file = $elf.file
        }
    }
}) | Sort-Object start
for ($index = 1; $index -lt $memoryRanges.Count; ++$index) {
    if ($memoryRanges[$index].start -lt $memoryRanges[$index - 1].end) {
        throw ('Overlapping ELF memory ranges: ' + $memoryRanges[$index - 1].file + ' / ' + $memoryRanges[$index].file)
    }
}
$symbols = Invoke-CaptureTool (Join-Path $a53Tools 'aarch64-none-elf-nm.exe') @('-n', $application)
if ($symbols -notmatch '(?m)^0*10000\s+A\s+_STACK_SIZE\s*$' -or
    $symbols -notmatch '(?m)^0*10000\s+A\s+_HEAP_SIZE\s*$') {
    throw 'The application must have a 64 KiB main stack and heap.'
}
$bootReadback = Invoke-CaptureTool (Join-Path $VitisRoot 'bin\bootgen.bat') @('-arch', 'zynqmp', '-read', $bootImage)
Set-Content -LiteralPath (Join-Path $artifacts 'bootgen_readback.txt') -Value $bootReadback
$applicationPartition = [regex]::Match($bootReadback,
    '(?s)PARTITION HEADER TABLE \(camera_capture\.elf\.0\)(.*)').Groups[1].Value
foreach ($expected in @(
    'exec_addr_lo \(0x10\)\s*:\s*0x00100000',
    'exec_addr_hi \(0x14\)\s*:\s*0x00000000',
    'load_addr_lo \(0x18\)\s*:\s*0x00100000',
    'load_addr_hi \(0x1c\)\s*:\s*0x00000000',
    'trustzone \[secure\]', 'el \[el-3\]', 'exec_state \[aarch-64\]',
    'dest_device \[PS\]', 'core \[a53-0\]'
)) {
    if ($applicationPartition -notmatch $expected) { throw "Unexpected application boot header: $expected" }
}
if ($bootReadback -notmatch 'fsbl_exec_address \(0x2c\)\s*:\s*0xfffc0000' -or
    $bootReadback -notmatch 'IMAGE HEADER \(camera_capture_wrapper\.bit\)' -or
    $bootReadback -notmatch 'dest_device \[PL\]' -or
    $bootReadback -notmatch 'pmufw_length \(0x34\)\s*:\s*(0x[0-9a-fA-F]+)' -or
    [Convert]::ToUInt32($Matches[1].Substring(2), 16) -eq 0) {
    throw 'BOOT.BIN must contain the FSBL, PMU firmware, PL bitstream and A53 application.'
}
$artifactHashes = @(Get-ChildItem -LiteralPath $artifacts -File | Where-Object { $_.Extension -in '.elf', '.bif', '.BIN', '.map' } | ForEach-Object {
    [ordered]@{ file = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName).Hash }
})
$report = [ordered]@{
    verifiedAt = Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK'
    toolchain = 'Vitis 2024.2 A53 GCC; -mcpu=cortex-a53 -std=c11 -O2 -g3 -Wall -Wextra -Werror'
    xsa = [ordered]@{ file = $xsa; sha256 = $xsaHash; embeddedBitstreamSha256 = $embeddedHash }
    hardwareValidationMarker = $markerText
    sourceCopiesMatch = $true
    elfMemoryRangesDoNotOverlap = $true
    validatedBootAttributes = 'FSBL entry 0xFFFC0000; PMU firmware present; PL bitstream; application load/exec 0x00100000, A53-0 AArch64 secure EL3'
    sourceHashes = $sourceHashes
    artifactHashes = $artifactHashes
    elf = $elfReports
    applicationSize = Invoke-CaptureTool (Join-Path $a53Tools 'aarch64-none-elf-size.exe') @($application)
    startupDisassembly = Invoke-CaptureTool (Join-Path $a53Tools 'aarch64-none-elf-objdump.exe') @('-d', '--disassemble=_startup', $application)
    relevantSymbols = @($symbols -split '\r?\n' | Where-Object { $_ -match '_startup|_vector_table|__bss|camera_frame|display_frames|_heap|__el3_stack|_STACK_SIZE' })
    bootgenReadback = $bootReadback
    physicalBoardTest = 'Not performed; CSI electrical link, HDMI output and SD card I/O still require the target board.'
}
$report | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $artifacts 'firmware_validation.json') -Encoding UTF8
Write-Host 'PASS: matching final XSA/bitstream/BSP/source copies, boot header readback, ELF ranges and 64 KiB stack/heap.'
