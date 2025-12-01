#!/bin/bash

echo "=========================================="
echo "Audio Deepfake Detector - Setup"
echo "=========================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo " Python 3 not found. Please install Python 3."
    exit 1
fi

echo " Python 3 found: $(python3 --version)"

if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo " Virtual environment created"
else
    echo " Virtual environment already exists"
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing dependencies (this may take a few minutes)..."
echo ""
pip install -r requirements.txt

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "Important: To use the project, activate the virtual environment first:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can:"
echo "  1. python scripts/download_datasets.py"
echo "  2. python scripts/quickstart.py"
echo "  3. python demo/app.py"
echo ""
echo "To deactivate the virtual environment later:"
echo "  deactivate"
echo ""
