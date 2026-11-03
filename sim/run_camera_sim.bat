@echo off
setlocal
if not defined VIVADO_ROOT set "VIVADO_ROOT=D:\Xlinix\Vivado\2024.2"
call "%VIVADO_ROOT%\settings64.bat"
if errorlevel 1 exit /b %errorlevel%
set "CAMERA_ROOT=%~dp0.."
if not exist "%CAMERA_ROOT%\build\camera_capture_sim" mkdir "%CAMERA_ROOT%\build\camera_capture_sim"
cd /d "%CAMERA_ROOT%\build\camera_capture_sim"
call "%VIVADO_ROOT%\bin\xvlog.bat" -sv "%CAMERA_ROOT%\src\rtl\camera\axis_frame_gate.v" "%CAMERA_ROOT%\src\rtl\camera\camera_isp_control.v" "%CAMERA_ROOT%\sim\rtl\tb_axis_frame_gate.sv"
if errorlevel 1 exit /b %errorlevel%
call "%VIVADO_ROOT%\bin\xelab.bat" tb_axis_frame_gate -s camera_frame_gate
if errorlevel 1 exit /b %errorlevel%
call "%VIVADO_ROOT%\bin\xsim.bat" camera_frame_gate -runall
exit /b %errorlevel%
