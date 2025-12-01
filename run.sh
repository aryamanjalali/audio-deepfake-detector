#!/bin/bash
# Quick run script - automatically activates venv and runs commands

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

source venv/bin/activate

# Run the command passed as arguments
"$@"
