#!/bin/bash
echo "==================================="
echo " Nullpk YT Automation Setup"
echo "==================================="

# Check for Python
if ! command -v python3 &> /dev/null
then
    echo "Python not found. Please install Python 3."
    exit
fi

# Create a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing required libraries..."
pip3 install -r requirements.txt

# Install PyInstaller
echo "Installing PyInstaller..."
pip3 install pyinstaller

# Run PyInstaller
echo "Creating the one-click executable..."
pyinstaller --onefile --windowed --name "Nullpk_YT_Automation" main.py

echo ""
echo "==================================="
echo " Setup Complete!"
echo "==================================="
echo "You can find the executable in the 'dist' folder."
echo "Look for 'Nullpk_YT_Automation'"
echo ""