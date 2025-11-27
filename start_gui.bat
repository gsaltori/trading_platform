@echo off
REM Trading Platform - GUI Launcher
REM ================================

echo.
echo =============================================
echo    TRADING PLATFORM - Strategy Generator
echo =============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.10 or higher
    pause
    exit /b 1
)

REM Launch the GUI
echo Starting Trading Platform GUI...
echo.
python run_gui.py

if errorlevel 1 (
    echo.
    echo =============================================
    echo    ERROR: GUI failed to start
    echo =============================================
    echo.
    echo If dependencies are missing, run:
    echo    install_dependencies.bat
    echo.
    pause
)
