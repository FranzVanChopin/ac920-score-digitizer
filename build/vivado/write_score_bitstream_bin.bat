@echo off
setlocal
if not defined VIVADO_ROOT set "VIVADO_ROOT=D:\Xlinix\Vivado\2024.2"
if not exist "%VIVADO_ROOT%\settings64.bat" exit /b 2
call "%VIVADO_ROOT%\settings64.bat"
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0..\.."
call "%VIVADO_ROOT%\bin\bootgen.bat" -arch zynqmp -image build/vivado/score_system_bitstream.bif -process_bitstream bin -w on
exit /b %errorlevel%
