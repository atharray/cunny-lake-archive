#!/bin/bash
# This script sets up a Python virtual environment, installs dependencies,
# and runs the main application for Linux/macOS.

echo "Running Python setup script..."

# Check if the 'venv' directory exists. If not, create the virtual environment.
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
# The 'source' command is used to run the activation script in the current shell.
# This is required for the virtual environment to be active for subsequent commands.
source venv/bin/activate

# Check if the activation was successful
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "ERROR: Failed to activate the virtual environment."
    echo "Please ensure Python 3 is installed and in your system's PATH."
    exit 1
fi

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the main Python application
echo "Running the application..."
python cunny.py

# The script will now exit
echo "Script finished."
exit 0
