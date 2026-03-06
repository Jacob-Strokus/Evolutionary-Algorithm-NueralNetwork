#!/bin/bash
# Quick launcher for EA-NN with GPU acceleration
# This script ensures we're in the right directory and uses the virtual environment

echo "🚀 EA-NN GPU Launcher"
echo "===================="

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
VENV_PATH="./scripts/ea-nn-gpu-env"
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run: ./scripts/gpu/setup_gpu.sh first"
    exit 1
fi

# Determine the correct Python executable
if [[ $(uname -r) == *microsoft* ]]; then
    # WSL - use Windows Python
    PYTHON_CMD="$VENV_PATH/Scripts/python.exe"
else
    # Linux
    PYTHON_CMD="$VENV_PATH/bin/python"
fi

echo "🔍 Using Python: $PYTHON_CMD"
echo "📁 Working directory: $(pwd)"
echo ""

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in current directory"
    echo "Current directory: $(pwd)"
    echo "Files: $(ls -la)"
    exit 1
fi

echo "🎮 Starting EA-NN with GPU acceleration..."
echo "🌐 Web interface will be available at: http://localhost:5000"
echo ""

# Run with all arguments passed to this script
$PYTHON_CMD main.py "$@"
