#!/usr/bin/env python3
"""
GPU Acceleration Test for EA-NN
===============================

Quick test to verify RTX 2060 GPU acceleration is working properly.
This script tests PyTorch CUDA functionality and EA-NN GPU integration.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_gpu_acceleration():
    """Test GPU acceleration capabilities"""
    print("🧪 EA-NN GPU Acceleration Test")
    print("=" * 35)
    
    # Test 1: PyTorch CUDA
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA Available: {cuda_available}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_version = torch.version.cuda
            
            print(f"   GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f}GB")
            print(f"   CUDA Version: {cuda_version}")
            
            # Test GPU tensor operations
            print("\n🔬 Testing GPU tensor operations...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"   Matrix multiplication: {z.shape} ✅")
            
        else:
            print("   ⚠️ CUDA not available - check installation")
            return False
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False
    
    # Test 2: EA-NN GPU Components
    try:
        print("\n🧠 Testing EA-NN GPU components...")
        
        # Test GPU detection in main
        from main import GPU_AVAILABLE, GPU_NAME, GPU_MEMORY
        print(f"   EA-NN GPU Detection: {GPU_AVAILABLE}")
        print(f"   Detected GPU: {GPU_NAME}")
        print(f"   Available Memory: {GPU_MEMORY}GB")
        
        # Test GPU accelerated environment
        from main import Phase2NeuralEnvironment
        env = Phase2NeuralEnvironment(width=50, height=50, use_gpu=True)
        print(f"   GPU Environment: {'✅' if env.use_gpu else '❌'}")
        print(f"   GPU Threshold: {env.gpu_threshold} agents")
        
        # Test GPU components
        if hasattr(env, 'gpu_neural_net'):
            print("   GPU Neural Network: ✅")
        if hasattr(env, 'gpu_spatial_ops'):
            print("   GPU Spatial Operations: ✅")
            
    except Exception as e:
        print(f"❌ EA-NN GPU component test failed: {e}")
        return False
    
    print("\n🎉 All GPU tests passed!")
    print("\n📊 Performance Expectations:")
    print("   • Small populations (<50): CPU optimized")
    print("   • Large populations (50+): GPU accelerated")
    print("   • Expected speedup: 3-5x on RTX 2060")
    print("\n🚀 Ready for GPU-accelerated simulations!")
    
    return True

def benchmark_gpu_vs_cpu():
    """Quick benchmark of GPU vs CPU performance"""
    try:
        import torch
        import time
        
        print("\n⚡ GPU vs CPU Benchmark")
        print("=" * 25)
        
        # Test data
        size = 2000
        
        # CPU benchmark
        print("🖥️ CPU Test...")
        start = time.time()
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start
        print(f"   CPU Time: {cpu_time:.3f}s")
        
        # GPU benchmark
        print("🎮 GPU Test...")
        start = time.time()
        x_gpu = torch.randn(size, size).cuda()
        y_gpu = torch.randn(size, size).cuda()
        torch.cuda.synchronize()  # Wait for GPU
        z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()  # Wait for completion
        gpu_time = time.time() - start
        print(f"   GPU Time: {gpu_time:.3f}s")
        
        speedup = cpu_time / gpu_time
        print(f"\n🏆 Speedup: {speedup:.1f}x faster on GPU!")
        
        if speedup > 2.0:
            print("✅ Excellent GPU acceleration!")
        elif speedup > 1.5:
            print("✅ Good GPU acceleration")
        else:
            print("⚠️ Limited GPU acceleration - check drivers")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

if __name__ == "__main__":
    print("🎮 Testing RTX 2060 GPU acceleration for EA-NN\n")
    
    success = test_gpu_acceleration()
    
    if success:
        benchmark_gpu_vs_cpu()
        print("\n🎯 Next step: Run './run_gpu.sh --web' to see it in action!")
    else:
        print("\n❌ GPU setup needs attention")
        print("💡 Try running: ./scripts/gpu/setup_gpu.sh")
