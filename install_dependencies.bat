@echo off
REM Trading Platform - Dependency Installer
REM ========================================

echo.
echo =============================================
echo    TRADING PLATFORM - Installing Dependencies
echo =============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

echo Installing core dependencies...
echo.

REM Core dependencies
pip install pandas numpy pyyaml joblib

REM Machine Learning
pip install scikit-learn

REM Technical Analysis
pip install ta

REM GUI
pip install PyQt6

REM Charts
pip install matplotlib

echo.
echo =============================================
echo.
echo Core dependencies installed!
echo.

set /p INSTALL_ML="Install ML extensions (XGBoost, LightGBM)? [y/N]: "
if /i "%INSTALL_ML%"=="y" (
    echo Installing ML extensions...
    pip install xgboost lightgbm
)

set /p INSTALL_MT5="Install MetaTrader5 connector? [y/N]: "
if /i "%INSTALL_MT5%"=="y" (
    echo Installing MetaTrader5...
    pip install MetaTrader5
)

echo.
echo =============================================
echo    Installation Complete!
echo =============================================
echo.
echo Run the platform with:
echo    start_gui.bat
echo.
echo Or:
echo    python run_gui.py
echo.

pause
