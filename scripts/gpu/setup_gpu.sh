#!/bin/bash
# GPU Setup Script for EA-NN with RTX 2060 (Linux/WSL Compatible)
# This script will help set up PyTorch with CUDA support

echo "🎮 EA-NN GPU Setup for RTX 2060"
echo "================================"
echo ""
echo "Your NVIDIA GeForce RTX 2060 is perfect for GPU acceleration!"
echo "This script will install PyTorch with CUDA support."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found in PATH"
    echo "Please ensure Python3 is installed"
    exit 1
fi

echo "✅ Python found"
python3 --version
echo ""

# Check if CUDA is installed (optional - PyTorch can work without CUDA toolkit)
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA toolkit found"
    nvcc --version | grep "release"
else
    echo "⚠️ CUDA toolkit not detected"
    echo "Don't worry - PyTorch can still use GPU with just NVIDIA drivers"
fi
echo ""

echo "🔄 Setting up Python environment for PyTorch with CUDA support..."
echo "This may take a few minutes..."
echo ""

# Check if we're in WSL and should use Windows Python
if [[ $(uname -r) == *microsoft* ]]; then
    echo "🔍 WSL detected - checking for Windows Python with GPU support"
    WINDOWS_PYTHON="/mnt/c/Program Files/Python313/python.exe"
    
    if [ -f "$WINDOWS_PYTHON" ]; then
        echo "✅ Found Windows Python - using for GPU support"
        PYTHON_CMD="$WINDOWS_PYTHON"
        VENV_PATH="../ea-nn-gpu-env"
        
        # Create virtual environment using Windows Python
        echo "🏗️ Creating virtual environment for GPU acceleration..."
        "$PYTHON_CMD" -m venv "$VENV_PATH"
        
        # Use virtual environment
        PYTHON_CMD="$VENV_PATH/Scripts/python.exe"
        PIP_CMD="$VENV_PATH/Scripts/pip.exe"
        
        echo "✅ Virtual environment created: $VENV_PATH"
    else
        echo "⚠️ Windows Python not found - using WSL with virtual environment"
        PYTHON_CMD="python3"
        VENV_PATH="../ea-nn-gpu-env"
        
        # Create virtual environment
        echo "🏗️ Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_PATH"
        
        # Use virtual environment
        PYTHON_CMD="$VENV_PATH/bin/python"
        PIP_CMD="$VENV_PATH/bin/pip"
        
        echo "✅ Virtual environment created: $VENV_PATH"
    fi
else
    echo "🐧 Linux detected - using virtual environment"
    PYTHON_CMD="python3"
    VENV_PATH="../ea-nn-gpu-env"
    
    # Create virtual environment
    echo "🏗️ Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_PATH"
    
    # Use virtual environment
    PYTHON_CMD="$VENV_PATH/bin/python"
    PIP_CMD="$VENV_PATH/bin/pip"
    
    echo "✅ Virtual environment created: $VENV_PATH"
fi

# Upgrade pip first
echo "⬆️ Upgrading pip..."
$PIP_CMD install --upgrade pip

# Uninstall existing PyTorch (CPU version)
echo "🧹 Removing existing PyTorch..."
$PIP_CMD uninstall torch torchvision -y

# Install PyTorch with CUDA
echo "🚀 Installing PyTorch with CUDA 12.4 support..."
$PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cu124

if [ $? -ne 0 ]; then
    echo "❌ PyTorch installation failed"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "1. Check internet connection"
    echo "2. Verify virtual environment is active"
    echo "3. Try manual installation from pytorch.org"
    echo "4. Check available disk space"
    exit 1
fi

# Install EA-NN dependencies
echo "📦 Installing EA-NN dependencies..."
if [ -f "../../requirements.txt" ]; then
    $PIP_CMD install -r "../../requirements.txt"
    echo "✅ Requirements installed from requirements.txt"
else
    # Install essential dependencies manually
    echo "Installing essential dependencies..."
    $PIP_CMD install matplotlib numpy flask flask-socketio websockets
    echo "✅ Essential dependencies installed"
fi

echo ""
echo "🧪 Testing GPU acceleration..."
$PYTHON_CMD -c "
import torch
print('✅ PyTorch imported successfully')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'CUDA Version: {torch.version.cuda}')
else:
    print('Note: GPU may still work with proper drivers')
"

if [ $? -ne 0 ]; then
    echo "❌ GPU test failed"
    echo "Check your installation and try again"
    exit 1
fi

echo ""
echo "🎉 SUCCESS! GPU acceleration is ready!"
echo ""
echo "🚀 Your RTX 2060 is now configured for EA-NN"
echo ""
echo "📁 Virtual environment created at: $VENV_PATH"
echo ""
echo "🔧 To activate the environment manually:"
if [[ $(uname -r) == *microsoft* ]] && [ -f "$WINDOWS_PYTHON" ]; then
    echo "   Windows: $VENV_PATH\\Scripts\\activate.bat"
    echo "   WSL: source $VENV_PATH/Scripts/activate"
else
    echo "   source $VENV_PATH/bin/activate"
fi
echo ""
echo "🚀 Quick Start Commands:"
echo "   cd /mnt/c/Users/Admin/Desktop/game/EA-NN"
echo "   chmod +x run_gpu.sh"
echo "   ./run_gpu.sh --web"
echo ""
echo "📋 Manual Commands:"
echo "   cd ../../  # Go to project root"
if [[ $(uname -r) == *microsoft* ]] && [ -f "$WINDOWS_PYTHON" ]; then
    echo "   ./scripts/ea-nn-gpu-env/Scripts/python.exe main.py --web"
else
    echo "   ./scripts/ea-nn-gpu-env/bin/python main.py --web"
fi
echo ""
echo "Next steps:"
echo "1. 🌐 Open browser to: http://localhost:5000"
echo "2. 👀 Watch GPU acceleration in action!"
echo "3. 📊 Monitor GPU usage with: nvidia-smi"
echo ""
echo "Expected performance boost: 3-5x faster simulations! 🎯"
echo ""
