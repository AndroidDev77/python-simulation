#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Ensure Python 3 is installed."
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing required libraries..."
pip install numpy matplotlib

if [ $? -ne 0 ]; then
    echo "Failed to install libraries. Check your internet connection or pip configuration."
    deactivate
    exit 1
fi

echo "Deactivating virtual environment..."
deactivate

echo "Setup complete. Use 'source venv/bin/activate' to activate the virtual environment."
