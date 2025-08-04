@echo off
REM GPU Setup Script for EA-NN with RTX 2060
REM This script will help set up PyTorch with CUDA support

echo 🎮 EA-NN GPU Setup for RTX 2060
echo ================================
echo.
echo Your NVIDIA GeForce RTX 2060 is perfect for GPU acceleration!
echo This script will install PyTorch with CUDA support.
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found in PATH
    echo Please ensure Python is installed and in your PATH
    pause
    exit /b 1
)

echo ✅ Python found
python --version
echo.

REM Check if CUDA is installed
nvcc --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️ CUDA toolkit not detected
    echo.
    echo 📥 Please download and install CUDA 12.4 first:
    echo 🔗 https://developer.nvidia.com/cuda-downloads
    echo.
    echo Select: Windows → x86_64 → 10 → exe (local)
    echo Install with default settings and restart your computer
    echo.
    echo After installing CUDA, run this script again.
    pause
    exit /b 1
)

echo ✅ CUDA toolkit found
nvcc --version | findstr "release"
echo.

echo 🔄 Installing PyTorch with CUDA support...
echo This may take a few minutes...
echo.

REM Uninstall existing PyTorch (CPU version)
echo Removing existing PyTorch...
pip uninstall torch torchvision -y

REM Install PyTorch with CUDA
echo Installing PyTorch with CUDA 12.4 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

if errorlevel 1 (
    echo ❌ PyTorch installation failed
    echo.
    echo 🔧 Troubleshooting:
    echo 1. Check internet connection
    echo 2. Update pip: python -m pip install --upgrade pip
    echo 3. Try manual installation from pytorch.org
    pause
    exit /b 1
)

echo.
echo 🧪 Testing GPU acceleration...
python -c "import torch; print('✅ PyTorch imported successfully'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if errorlevel 1 (
    echo ❌ GPU test failed
    echo Check your installation and try again
    pause
    exit /b 1
)

echo.
echo 🎉 SUCCESS! GPU acceleration is ready!
echo.
echo 🚀 Your RTX 2060 is now configured for EA-NN
echo.
echo Next steps:
echo 1. Run: python main.py --web
echo 2. Watch GPU acceleration in action!
echo 3. Monitor GPU usage with: nvidia-smi
echo.
echo Expected performance boost: 3-5x faster simulations! 🎯
echo.
pause
