#!/bin/bash

echo "====================================="
echo "  Tile Stitcher - Unix Launcher"
echo "====================================="

read -p "This will create a virtual environment (venv). Continue? (y/n) " CONTINUE
if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
    echo "Exiting..."
    exit 0
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Make sure Python3 is installed."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    read -p "Press Enter to exit..."
    exit 1
fi

python -m pip show requests > /dev/null 2>&1
if [ $? -ne 0 ]; then
    read -p "Requests missing. Install requirements.txt now? (y/n) " INSTALL
    if [[ "$INSTALL" == "y" || "$INSTALL" == "Y" ]]; then
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Failed to install requirements."
            read -p "Press Enter to exit..."
            exit 1
        fi
    else
        echo "Cannot continue without requests. Exiting..."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

python -m pip show Pillow > /dev/null 2>&1
if [ $? -ne 0 ]; then
    read -p "Pillow missing. Install requirements.txt now? (y/n) " INSTALL2
    if [[ "$INSTALL2" == "y" || "$INSTALL2" == "Y" ]]; then
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Failed to install requirements."
            read -p "Press Enter to exit..."
            exit 1
        fi
    else
        echo "Cannot continue without Pillow. Exiting..."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

echo "Starting Tile Stitcher..."
python stitch_tiles.py
if [ $? -ne 0 ]; then
    echo "Script execution failed."
fi

read -p "Press Enter to exit..."
