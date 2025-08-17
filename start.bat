@echo off
setlocal enabledelayedexpansion
echo =====================================
echo   Tile Stitcher - Windows Launcher
echo =====================================
set "CONTINUE="
set /p CONTINUE=This will create a virtual environment (venv). Continue? (y/n) 
if /I "%CONTINUE%" NEQ "y" (
    echo Exiting...
    exit /b
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if !ERRORLEVEL! NEQ 0 (
        echo Failed to create virtual environment. Make sure Python is installed.
        pause
        exit /b
    )
)

call venv\Scripts\activate
if !ERRORLEVEL! NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b
)

python -m pip show requests >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    set "INSTALL="
    set /p INSTALL=Requests missing. Install requirements.txt now? (y/n) 
    if /I "!INSTALL!"=="y" (
        pip install -r requirements.txt
        if !ERRORLEVEL! NEQ 0 (
            echo Failed to install requirements.
            pause
            exit /b
        )
    ) else (
        echo Cannot continue without requests. Exiting...
        pause
        exit /b
    )
)

python -m pip show Pillow >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    set "INSTALL2="
    set /p INSTALL2=Pillow missing. Install requirements.txt now? (y/n) 
    if /I "!INSTALL2!"=="y" (
        pip install -r requirements.txt
        if !ERRORLEVEL! NEQ 0 (
            echo Failed to install requirements.
            pause
            exit /b
        )
    ) else (
        echo Cannot continue without Pillow. Exiting...
        pause
        exit /b
    )
)

echo Starting Tile Stitcher...
python stitch_tiles.py
if !ERRORLEVEL! NEQ 0 (
    echo Script execution failed.
)
pause
