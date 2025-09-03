@echo off
echo ===================================
echo  Nullpk YT Automation Setup
echo ===================================

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install it from python.org and add it to your PATH.
    pause
    exit
)

REM Create a virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo Installing required libraries...
pip install -r requirements.txt

REM Install PyInstaller for packaging
echo Installing PyInstaller...
pip install pyinstaller

REM Run PyInstaller to create the executable
echo Creating the one-click executable (this may take a few moments)...
pyinstaller --onefile --windowed --name "Nullpk_YT_Automation" main.py

echo.
echo ===================================
echo  Setup Complete!
echo ===================================
echo You can find the executable file in the 'dist' folder.
echo Look for 'Nullpk_YT_Automation.exe'
echo.
pause