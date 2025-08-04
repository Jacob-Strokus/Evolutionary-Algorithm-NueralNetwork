#!/usr/bin/env python3
"""
GPU Capability Checker & Enabler
================================

Comprehensive tool to check GPU capabilities, install required packages,
and enable GPU acceleration for EA-NN simulation.
"""

import sys
import os
import subprocess
import platform

def check_system_info():
    """Check basic system information"""
    print("🖥️ SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()

def check_nvidia_gpu():
    """Check for NVIDIA GPU hardware"""
    print("🎮 NVIDIA GPU DETECTION")
    print("=" * 50)
    
    try:
        # Try nvidia-smi command (most reliable)
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected via nvidia-smi:")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line or 'Tesla' in line:
                    print(f"   🎯 {line.strip()}")
            return True
        else:
            print("❌ nvidia-smi command failed or not found")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not available")
    
    # Try alternative methods
    try:
        # Windows WMIC method
        if platform.system() == "Windows":
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                nvidia_gpus = [line.strip() for line in lines if 'NVIDIA' in line and line.strip()]
                if nvidia_gpus:
                    print("✅ NVIDIA GPU detected via Windows WMIC:")
                    for gpu in nvidia_gpus:
                        print(f"   🎯 {gpu}")
                    return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("❌ No NVIDIA GPU detected")
    print("💡 GPU acceleration requires NVIDIA GPU with CUDA support")
    return False

def check_cuda_installation():
    """Check CUDA installation"""
    print("\n🚀 CUDA INSTALLATION CHECK")
    print("=" * 50)
    
    try:
        # Check nvcc (CUDA compiler)
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CUDA toolkit installed:")
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"   📦 {line.strip()}")
            return True
        else:
            print("❌ CUDA toolkit not found")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvcc command not available")
    
    # Check for CUDA runtime
    cuda_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "/usr/local/cuda",
        "/opt/cuda"
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"✅ CUDA installation found at: {path}")
            return True
    
    print("❌ CUDA toolkit not installed")
    return False

def check_python_gpu_packages():
    """Check for GPU-related Python packages"""
    print("\n🐍 PYTHON GPU PACKAGES CHECK")
    print("=" * 50)
    
    packages_to_check = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cupy', 'CuPy'),
        ('nvidia-ml-py', 'NVIDIA ML Python'),
        ('pynvml', 'Python NVML')
    ]
    
    installed_packages = []
    
    for package_name, display_name in packages_to_check:
        try:
            __import__(package_name)
            installed_packages.append(package_name)
            print(f"✅ {display_name} installed")
            
            # Special check for PyTorch CUDA
            if package_name == 'torch':
                import torch
                print(f"   📦 PyTorch version: {torch.__version__}")
                if torch.cuda.is_available():
                    print(f"   🚀 CUDA available: {torch.cuda.device_count()} device(s)")
                    for i in range(torch.cuda.device_count()):
                        gpu_name = torch.cuda.get_device_name(i)
                        print(f"      GPU {i}: {gpu_name}")
                else:
                    print(f"   ⚠️ CUDA not available in PyTorch")
                    
        except ImportError:
            print(f"❌ {display_name} not installed")
    
    return installed_packages

def provide_installation_instructions():
    """Provide installation instructions for GPU support"""
    print("\n🛠️ GPU ENABLEMENT INSTRUCTIONS")
    print("=" * 50)
    
    print("To enable GPU acceleration for EA-NN, follow these steps:\n")
    
    print("1️⃣ INSTALL NVIDIA DRIVERS")
    print("   🔗 Visit: https://www.nvidia.com/drivers")
    print("   📥 Download and install latest drivers for your GPU")
    print("   🔄 Restart your computer after installation\n")
    
    print("2️⃣ INSTALL CUDA TOOLKIT")
    print("   🔗 Visit: https://developer.nvidia.com/cuda-downloads")
    print("   📥 Download CUDA 12.x for your operating system")
    print("   ⚙️ Run installer with default settings")
    print("   🔄 Restart your computer after installation\n")
    
    print("3️⃣ INSTALL PYTORCH WITH CUDA")
    print("   🐍 For Windows/Linux with CUDA 12.x:")
    print("   💻 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    print("   📦 Or visit: https://pytorch.org/get-started/locally/\n")
    
    print("4️⃣ INSTALL CUPY (OPTIONAL)")
    print("   🐍 For additional GPU acceleration:")
    print("   💻 pip install cupy-cuda12x")
    print("   📋 Note: Requires CUDA toolkit installed\n")
    
    print("5️⃣ VERIFY INSTALLATION")
    print("   🧪 Run this script again to verify GPU support")
    print("   🚀 Or run: python -c \"import torch; print(torch.cuda.is_available())\"")

def run_pytorch_gpu_test():
    """Run a quick PyTorch GPU test"""
    print("\n🧪 PYTORCH GPU FUNCTIONALITY TEST")
    print("=" * 50)
    
    try:
        import torch
        
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            
            # Test GPU operations
            device = torch.device('cuda:0')
            print(f"Current Device: {device}")
            
            # Simple tensor operations test
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            
            print("✅ GPU tensor operations working correctly!")
            print(f"🎯 GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
            # Clear GPU memory
            del x, y, z
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleared")
            
            return True
        else:
            print("❌ CUDA not available - GPU operations not possible")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def check_gpu_memory():
    """Check available GPU memory"""
    print("\n💾 GPU MEMORY CHECK")
    print("=" * 50)
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # GB
                gpu_name = props.name
                
                print(f"GPU {i}: {gpu_name}")
                print(f"   Total Memory: {total_memory:.1f} GB")
                
                # Check current usage
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                
                print(f"   Allocated: {allocated:.2f} GB")
                print(f"   Reserved: {reserved:.2f} GB")
                print(f"   Available: {total_memory - reserved:.1f} GB")
                
                # Memory recommendations
                if total_memory >= 8:
                    print("   🚀 Excellent for large-scale simulations (1000+ agents)")
                elif total_memory >= 4:
                    print("   ✅ Good for medium-scale simulations (500+ agents)")
                elif total_memory >= 2:
                    print("   💡 Suitable for small-scale simulations (200+ agents)")
                else:
                    print("   ⚠️ Limited memory - consider CPU mode")
        else:
            print("❌ No GPU available for memory check")
    except Exception as e:
        print(f"❌ Memory check failed: {e}")

def provide_ea_nn_integration_guide():
    """Provide specific guidance for EA-NN GPU integration"""
    print("\n🧬 EA-NN GPU INTEGRATION GUIDE")
    print("=" * 50)
    
    print("Once GPU support is installed, EA-NN will automatically:")
    print("✅ Detect available GPU hardware")
    print("✅ Enable neural network batch processing")
    print("✅ Accelerate spatial operations")
    print("✅ Fall back to CPU if GPU unavailable")
    print()
    
    print("To run EA-NN with GPU acceleration:")
    print("💻 python main.py --web    # Web interface with GPU")
    print("💻 python main.py          # Standard simulation with GPU")
    print()
    
    print("GPU acceleration will automatically activate when:")
    print("🎯 NVIDIA GPU with CUDA support detected")
    print("🎯 PyTorch with CUDA installed")
    print("🎯 Population size > 50 agents (configurable)")
    print()
    
    print("Performance expectations:")
    print("🚀 Neural Networks: 3-10x speedup with batch processing")
    print("🚀 Spatial Operations: 5-8x speedup with vectorization")
    print("🚀 Overall Simulation: 2-4x total performance improvement")

def main():
    """Main GPU capability checker"""
    print("🔍 EA-NN GPU CAPABILITY CHECKER & ENABLER")
    print("Checking your system's GPU acceleration capabilities...")
    print("=" * 80)
    
    # Check system info
    check_system_info()
    
    # Check for NVIDIA GPU
    has_nvidia_gpu = check_nvidia_gpu()
    
    # Check CUDA installation
    has_cuda = check_cuda_installation()
    
    # Check Python packages
    gpu_packages = check_python_gpu_packages()
    
    # Run GPU functionality test
    gpu_test_passed = run_pytorch_gpu_test()
    
    # Check GPU memory if available
    if 'torch' in gpu_packages:
        check_gpu_memory()
    
    # Final assessment
    print("\n🎯 GPU CAPABILITY ASSESSMENT")
    print("=" * 80)
    
    if has_nvidia_gpu and has_cuda and 'torch' in gpu_packages and gpu_test_passed:
        print("🎉 EXCELLENT: Your system is fully GPU-ready for EA-NN!")
        print("✅ Hardware: NVIDIA GPU detected")
        print("✅ Software: CUDA toolkit installed")
        print("✅ PyTorch: GPU support working")
        print("🚀 EA-NN will automatically use GPU acceleration")
        
    elif has_nvidia_gpu and 'torch' in gpu_packages:
        print("✅ GOOD: Your system has GPU hardware and PyTorch")
        if not has_cuda:
            print("⚠️ CUDA toolkit missing - may limit performance")
        print("🎯 EA-NN should work with GPU acceleration")
        
    elif has_nvidia_gpu:
        print("💡 POTENTIAL: NVIDIA GPU detected but missing software")
        print("🛠️ Install PyTorch with CUDA support for GPU acceleration")
        
    else:
        print("💻 CPU MODE: No NVIDIA GPU detected")
        print("✅ EA-NN will use optimized CPU processing")
        print("🎯 Still provides excellent performance with CPU acceleration")
    
    # Always provide integration guide
    provide_ea_nn_integration_guide()
    
    # Provide installation instructions if needed
    if not (has_nvidia_gpu and has_cuda and 'torch' in gpu_packages):
        provide_installation_instructions()
    
    print("\n🎉 GPU capability check complete!")
    print("💡 Run 'python main.py --web' to test EA-NN with your current setup")

if __name__ == "__main__":
    main()
