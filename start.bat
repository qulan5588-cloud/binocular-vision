@echo off
rem ---------------------------------------------------------------------------
rem  start.bat  --  double-click to open the main UI (app_launcher.py cards)
rem  usage:  start.bat            -> main UI
rem          start.bat menu       -> old text menu (scan / voice / check ...)
rem          start.bat scan^|tsdf^|voice^|both^|check^|list
rem  all real logic lives in _launch.py (venv pick / memory check / Chinese UI)
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [warn] .venv not found, falling back to python on PATH
    set "PY=python"
)

"%PY%" "%~dp0_launch.py" %*
set "RC=%ERRORLEVEL%"

rem only hold the window when something failed, so closing the main UI is clean
if not "%RC%"=="0" pause
exit /b %RC%
