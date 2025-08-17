@echo off
echo Running Python setup script...

REM Check for an existing virtual environment and create it if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
echo Activating virtual environment...
REM The 'call' command is used to run the activation script and return to the main script
call venv\Scripts\activate.bat

REM Check if the activation was successful
if not defined VIRTUAL_ENV (
    echo.
    echo ERROR: Failed to activate the virtual environment.
    echo Please ensure Python is in your system's PATH.
    pause
    exit /b 1
)

REM Install dependencies from requirements.txt
echo Installing dependencies...
pip install -r requirements.txt

REM Run the main Python application
echo Running the application...
python cunny.py

REM The script will now exit
echo Script finished.
exit /b 0
