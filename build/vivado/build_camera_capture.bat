@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_camera_capture.ps1" %*
exit /b %errorlevel%
