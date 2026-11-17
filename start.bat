@echo off
rem ---------------------------------------------------------------------------
rem  start.bat  --  double-click to open the launcher menu
rem  usage:  start.bat [scan^|tsdf^|voice^|both^|toolbox^|check^|list]
rem  all real logic lives in _launch.py (Chinese menu / env checks there)
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

if "%~1"=="" pause
if not "%RC%"=="0" pause
exit /b %RC%
