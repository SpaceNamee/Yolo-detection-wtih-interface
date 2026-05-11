@echo off
setlocal

echo === YOLO App Setup ===
echo.

REM --- 1. Check that Python is installed ---
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Install Python 3.9+ from https://www.python.org/downloads/
    echo During install, tick "Add Python to PATH".
    pause
    exit /b 1
)

REM --- 2. Create venv if not already present ---
if exist ".venv\" (
    echo Virtual environment .venv already exists. Skipping creation.
) else (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

REM --- 3. Activate venv and install dependencies ---
echo Activating venv and installing packages...
call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip
if errorlevel 1 goto fail

pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo  To activate the environment later:
echo     .venv\Scripts\activate
echo.
echo  To run the app:
echo     python app.py
echo ============================================
pause
exit /b 0

:fail
echo [ERROR] Installation failed. See messages above.
pause
exit /b 1
