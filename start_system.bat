@echo off
title Threat Hunting Agent System

echo ============================================================
echo   AUTONOMOUS THREAT HUNTING AGENT
echo ============================================================
echo.
echo Starting system components...
echo.

cd F:\Projects\threat-hunting-agent

REM Start Dashboard
echo [1/2] Starting Dashboard...
start "Dashboard" cmd /k "call venv\Scripts\activate.bat && python dashboard_flask.py"
timeout /t 3 /nobreak >nul

REM Start Agent Monitor
echo [2/2] Starting Agent Monitor...
start "Agent Monitor" cmd /k "call venv\Scripts\activate.bat && python src\agent_monitor.py"

echo.
echo ============================================================
echo   SYSTEM STARTED!
echo ============================================================
echo.
echo Dashboard:      http://localhost:5000
echo Agent Monitor:  Running in separate window
echo.
echo Press any key to exit...
pause >nul
