# Run from CMD with: powershell -NoProfile -ExecutionPolicy Bypass -File "...\monitor_serial.ps1" -Port COM4
# Receives and logs board output. Typed characters are sent to the board.
param(
    [string]$Port = 'COM4',
    [ValidateRange(1, 4000000)][int]$BaudRate = 115200,
    [string]$LogPath,
    [string]$LogDirectory,
    [switch]$LogSelfTest
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $captureRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    $captureLogDir = Join-Path $captureRoot 'build\camera_serial_logs'
    if (-not [string]::IsNullOrWhiteSpace($LogDirectory)) { $captureLogDir=[IO.Path]::GetFullPath($LogDirectory) }
    [IO.Directory]::CreateDirectory($captureLogDir) | Out-Null
    $LogPath = Join-Path $captureLogDir ('serial_{0}_{1}.txt' -f $Port, (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
elseif (-not [string]::IsNullOrWhiteSpace($LogDirectory)) { throw 'Choose LogPath or LogDirectory, not both.' }
$LogPath = [IO.Path]::GetFullPath($LogPath)
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($LogPath)) | Out-Null
$captureSerial = New-Object System.IO.Ports.SerialPort $Port, $BaudRate, 'None', 8, 'One'
$captureSerial.Handshake = 'None'
$captureSerial.DtrEnable = $false
$captureSerial.RtsEnable = $false
$captureSerial.WriteTimeout = 500
$captureLogStream = $null
$captureLogWriter = $null
function Write-CaptureLog([string]$Text) {
    $captureLogWriter.Write($Text)
    $captureLogWriter.Flush()
    # Persist before printing: selecting/freezing a console must not hide the
    # most recently received chunk from the on-disk diagnostic log.
    $captureLogStream.Flush($true)
}
try {
    # CreateNew prevents accidental replacement of an earlier diagnostic log.
    $captureLogStream = [IO.FileStream]::new($LogPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write,
        [IO.FileShare]::Read, 4096, [IO.FileOptions]::WriteThrough)
    $captureLogWriter = [IO.StreamWriter]::new($captureLogStream, [Text.UTF8Encoding]::new($false))
    $captureLogWriter.AutoFlush = $true
    Write-CaptureLog ("[HOST START {0}] port={1} baud={2} log={3}`r`n" -f [DateTimeOffset]::Now.ToString('o'),$Port,$BaudRate,$LogPath)
    if ($LogSelfTest) {
        Write-CaptureLog "[HOST SELF TEST] local file only; no serial port opened.`r`n"
        $captureReader=[IO.StreamReader]::new([IO.File]::Open($LogPath,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite))
        try { $captureReadback=$captureReader.ReadToEnd() } finally { $captureReader.Dispose() }
        if ($captureReadback -notmatch 'HOST START' -or $captureReadback -notmatch 'HOST SELF TEST') { throw 'Log readback failed.' }
        Write-Host "PASS log live readback: $($captureLogStream.Length) bytes; $LogPath"
        return
    }
    $captureSerial.Open()
    Write-Host "Listening on $Port ($BaudRate 8N1). Ctrl+C to stop."
    Write-Host "Log ready ($($captureLogStream.Length) bytes): $LogPath"
    Write-Host 'After READY: f=focus on/off; 1-5=region, z=zoom, r=peak reset, g=five-region snapshot. No Enter needed.'
    Write-Host 'p/b/space=Image+Grey+Meta, t=next song, u=Ubuntu, s=status+DP, n=denoise, [ ]=exposure; outside focus: w=WB, a=AE on/lock.'
    Write-Host 'For low-ISO stills use AE ON; do not use [ ] (locks AE). RX and timestamped TX are saved locally.'
    while ($true) {
        $captureText = $captureSerial.ReadExisting()
        if ($captureText.Length -gt 0) {
            Write-CaptureLog $captureText
            [Console]::Write($captureText)
        }
        if (-not [Console]::IsInputRedirected -and [Console]::KeyAvailable) {
            $captureKey = [Console]::ReadKey($true)
            if ([int]$captureKey.KeyChar -ne 0) {
                Write-CaptureLog ("`r`n[HOST TX {0}] key=U+{1:X4} name={2}`r`n" -f [DateTimeOffset]::Now.ToString('o'),[int]$captureKey.KeyChar,$captureKey.Key)
                $captureSerial.Write([string]$captureKey.KeyChar)
            }
        }
        Start-Sleep -Milliseconds 20
    }
}
catch {
    if ($null -ne $captureLogWriter) {
        Write-CaptureLog ("`r`n[HOST ERROR {0}] {1}`r`n" -f [DateTimeOffset]::Now.ToString('o'),$_.Exception.Message)
    }
    throw
}
finally {
    $captureSerial.Dispose()
    if ($null -ne $captureLogWriter) {
        try { Write-CaptureLog ("`r`n[HOST STOP {0}]`r`n" -f [DateTimeOffset]::Now.ToString('o')) }
        finally { $captureLogWriter.Dispose() }
    }
    elseif ($null -ne $captureLogStream) { $captureLogStream.Dispose() }
}
