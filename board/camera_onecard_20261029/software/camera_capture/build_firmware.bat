@echo off
setlocal
for %%I in ("%~dp0..\..") do set "ONECARD_ROOT=%%~fI"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ONECARD_ROOT%\build_all.ps1" %*
exit /b %ERRORLEVEL%
