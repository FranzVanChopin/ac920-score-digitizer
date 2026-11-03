@echo off
setlocal
if not defined VIVADO_ROOT set "VIVADO_ROOT=D:\Xlinix\Vivado\2024.2"
if not exist "%VIVADO_ROOT%\settings64.bat" (
  echo Vivado settings64.bat not found: "%VIVADO_ROOT%\settings64.bat"
  echo Set VIVADO_ROOT to the Vivado installation directory and retry.
  exit /b 2
)
call "%VIVADO_ROOT%\settings64.bat"
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0..\.."

call "%VIVADO_ROOT%\bin\vivado.bat" -mode batch -nolog -nojournal -source build/vivado/create_score_system.tcl
if errorlevel 1 exit /b %errorlevel%
set "RUN_ROOT=%CD%\build\vivado_score_system\score_system.runs"

call :run_ooc score_system_ps_0_synth_1 score_system_ps_0.tcl
if errorlevel 1 exit /b %errorlevel%
call :run_ooc score_system_display_0_synth_1 score_system_display_0.tcl
if errorlevel 1 exit /b %errorlevel%
call :run_ooc score_system_axi_smc_0_synth_1 score_system_axi_smc_0.tcl
if errorlevel 1 exit /b %errorlevel%
call :run_ooc score_system_pl_reset_sync_0_synth_1 score_system_pl_reset_sync_0.tcl
if errorlevel 1 exit /b %errorlevel%
call :run_ooc score_system_resetn_invert_0_synth_1 score_system_resetn_invert_0.tcl
if errorlevel 1 exit /b %errorlevel%

pushd "%RUN_ROOT%\synth_1"
call "%VIVADO_ROOT%\bin\vivado.bat" -mode batch -nolog -nojournal -source score_system_wrapper.tcl
set "RUN_RESULT=%errorlevel%"
popd
if not "%RUN_RESULT%"=="0" exit /b %RUN_RESULT%

call "%VIVADO_ROOT%\bin\vivado.bat" -mode batch -nolog -nojournal -source build/vivado/generate_score_impl.tcl
if errorlevel 1 exit /b %errorlevel%
pushd "%RUN_ROOT%\impl_1"
call "%VIVADO_ROOT%\bin\vivado.bat" -mode batch -nolog -nojournal -source score_system_wrapper.tcl
set "RUN_RESULT=%errorlevel%"
popd
if not "%RUN_RESULT%"=="0" exit /b %RUN_RESULT%

call "%VIVADO_ROOT%\bin\vivado.bat" -mode batch -nolog -nojournal -source build/vivado/report_score_system.tcl
exit /b %errorlevel%

:run_ooc
pushd "%RUN_ROOT%\%~1"
call "%VIVADO_ROOT%\bin\vivado.bat" -mode batch -nolog -nojournal -source "%~2"
set "RUN_RESULT=%errorlevel%"
popd
if not "%RUN_RESULT%"=="0" exit /b %RUN_RESULT%
exit /b 0
