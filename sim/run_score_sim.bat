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
cd /d "%~dp0\.."

call "%VIVADO_ROOT%\bin\xvlog.bat" -sv ^
  src/rtl/score_event_player.sv src/rtl/axi_score_event_player.sv ^
  sim/rtl/tb_score_event_player.sv sim/rtl/tb_score_event_player_100mhz.sv ^
  sim/rtl/tb_axi_score_event_player.sv ^
  src/rtl/score_video_core.sv sim/rtl/tb_score_video_core.sv ^
  src/rtl/score_video_axi_control.sv sim/rtl/tb_score_video_axi_control.sv
if errorlevel 1 exit /b %errorlevel%

call "%VIVADO_ROOT%\bin\xelab.bat" tb_score_event_player -s sim_tone
if errorlevel 1 exit /b %errorlevel%
call "%VIVADO_ROOT%\bin\xsim.bat" sim_tone -tclbatch build/xvlog/run_score_sim.tcl
if errorlevel 1 exit /b %errorlevel%

call "%VIVADO_ROOT%\bin\xelab.bat" tb_score_event_player_100mhz -s sim_tone_100mhz
if errorlevel 1 exit /b %errorlevel%
call "%VIVADO_ROOT%\bin\xsim.bat" sim_tone_100mhz -tclbatch build/xvlog/run_score_sim.tcl
if errorlevel 1 exit /b %errorlevel%

call "%VIVADO_ROOT%\bin\xelab.bat" tb_axi_score_event_player -s sim_axi_audio
if errorlevel 1 exit /b %errorlevel%
call "%VIVADO_ROOT%\bin\xsim.bat" sim_axi_audio -tclbatch build/xvlog/run_score_sim.tcl
if errorlevel 1 exit /b %errorlevel%

call "%VIVADO_ROOT%\bin\xelab.bat" tb_score_video_core -s sim_video
if errorlevel 1 exit /b %errorlevel%
call "%VIVADO_ROOT%\bin\xsim.bat" sim_video -tclbatch build/xvlog/run_score_sim.tcl
if errorlevel 1 exit /b %errorlevel%

call "%VIVADO_ROOT%\bin\xelab.bat" tb_score_video_axi_control -s sim_video_axi
if errorlevel 1 exit /b %errorlevel%
call "%VIVADO_ROOT%\bin\xsim.bat" sim_video_axi -tclbatch build/xvlog/run_score_sim.tcl
exit /b %errorlevel%
