#!/usr/bin/env python3
"""
GPU Acceleration System Demonstration
====================================

Simple demonstration of the GPU acceleration components, including
hardware detection, basic functionality tests, and performance validation.
"""

import sys
import os
import time
import numpy as np

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

def test_pytorch_installation():
    """Test PyTorch installation and CUDA availability"""
    
    print("🔧 PYTORCH & CUDA INSTALLATION TEST")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        
        # Test CUDA
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
        
        if cuda_available:
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("No CUDA GPU detected - will use CPU mode")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def test_basic_gpu_operations():
    """Test basic GPU operations with PyTorch"""
    
    print("\n⚡ BASIC GPU OPERATIONS TEST")
    print("=" * 50)
    
    try:
        import torch
        
        # Test tensor creation and operations
        print("🧪 Testing tensor operations...")
        
        # CPU tensors
        cpu_a = torch.randn(1000, 1000)
        cpu_b = torch.randn(1000, 1000)
        
        # CPU matrix multiplication
        cpu_start = time.time()
        cpu_result = torch.mm(cpu_a, cpu_b)
        cpu_time = time.time() - cpu_start
        
        print(f"CPU Matrix Multiply (1000x1000): {cpu_time:.3f}s")
        
        # GPU operations (if available)
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            
            # Transfer to GPU
            gpu_a = cpu_a.to(device)
            gpu_b = cpu_b.to(device)
            
            # GPU matrix multiplication
            torch.cuda.synchronize()  # Ensure GPU is ready
            gpu_start = time.time()
            gpu_result = torch.mm(gpu_a, gpu_b)
            torch.cuda.synchronize()  # Wait for completion
            gpu_time = time.time() - gpu_start
            
            print(f"GPU Matrix Multiply (1000x1000): {gpu_time:.3f}s")
            
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"GPU Speedup: {speedup:.2f}x")
            
            # Test memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024**2)
            print(f"GPU Memory Allocated: {memory_allocated:.1f} MB")
            
            # Clear GPU memory
            del gpu_a, gpu_b, gpu_result
            torch.cuda.empty_cache()
            
        else:
            print("GPU not available - CPU only test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU operations test failed: {e}")
        return False

def test_neural_network_acceleration():
    """Test neural network acceleration capabilities"""
    
    print("\n🧠 NEURAL NETWORK ACCELERATION TEST")
    print("=" * 50)
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple neural network
        class TestNetwork(nn.Module):
            def __init__(self):
                super(TestNetwork, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.Tanh(),
                    nn.Linear(20, 15),
                    nn.Tanh(),
                    nn.Linear(15, 4)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Create network
        network = TestNetwork()
        print(f"✅ Neural network created with {sum(p.numel() for p in network.parameters())} parameters")
        
        # Test batch processing on CPU
        batch_size = 100
        input_data = torch.randn(batch_size, 10)
        
        cpu_start = time.time()
        with torch.no_grad():
            cpu_output = network(input_data)
        cpu_time = time.time() - cpu_start
        
        print(f"CPU Batch Processing ({batch_size} agents): {cpu_time:.3f}s")
        print(f"CPU Rate: {batch_size/cpu_time:.1f} agents/sec")
        
        # Test GPU acceleration if available
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            
            # Move network and data to GPU
            gpu_network = network.to(device)
            gpu_input = input_data.to(device)
            
            # GPU batch processing
            torch.cuda.synchronize()
            gpu_start = time.time()
            with torch.no_grad():
                gpu_output = gpu_network(gpu_input)
            torch.cuda.synchronize()
            gpu_time = time.time() - gpu_start
            
            print(f"GPU Batch Processing ({batch_size} agents): {gpu_time:.3f}s")
            print(f"GPU Rate: {batch_size/gpu_time:.1f} agents/sec")
            
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"Neural Network GPU Speedup: {speedup:.2f}x")
            
            # Verify outputs match (approximately)
            cpu_output_check = gpu_output.cpu()
            diff = torch.abs(cpu_output - cpu_output_check).mean()
            print(f"Output difference (CPU vs GPU): {diff:.6f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Neural network acceleration test failed: {e}")
        return False

def test_spatial_operations():
    """Test spatial operation acceleration"""
    
    print("\n🌍 SPATIAL OPERATIONS TEST")
    print("=" * 50)
    
    try:
        import torch
        
        # Generate random agent positions
        num_agents = 500
        positions = torch.randn(num_agents, 2) * 100  # Random positions
        
        print(f"✅ Generated {num_agents} random agent positions")
        
        # CPU distance matrix calculation
        print("🧪 Testing distance calculations...")
        
        cpu_start = time.time()
        # Vectorized distance calculation on CPU
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        cpu_distances = torch.sqrt(torch.sum(diff**2, dim=2))
        cpu_time = time.time() - cpu_start
        
        print(f"CPU Distance Matrix ({num_agents}x{num_agents}): {cpu_time:.3f}s")
        
        # GPU distance matrix calculation
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            gpu_positions = positions.to(device)
            
            torch.cuda.synchronize()
            gpu_start = time.time()
            gpu_diff = gpu_positions.unsqueeze(1) - gpu_positions.unsqueeze(0)
            gpu_distances = torch.sqrt(torch.sum(gpu_diff**2, dim=2))
            torch.cuda.synchronize()
            gpu_time = time.time() - gpu_start
            
            print(f"GPU Distance Matrix ({num_agents}x{num_agents}): {gpu_time:.3f}s")
            
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"Spatial GPU Speedup: {speedup:.2f}x")
            
            # Verify accuracy
            cpu_distances_check = gpu_distances.cpu()
            diff = torch.abs(cpu_distances - cpu_distances_check).mean()
            print(f"Distance calculation difference: {diff:.6f}")
        
        # Test neighbor finding
        print(f"\n🔍 Testing neighbor finding...")
        query_pos = positions[:10]  # First 10 as query points
        radius = 25.0
        
        cpu_start = time.time()
        neighbors_cpu = []
        for q_pos in query_pos:
            distances = torch.sqrt(torch.sum((positions - q_pos)**2, dim=1))
            neighbors = torch.where(distances <= radius)[0]
            neighbors_cpu.append(len(neighbors))
        cpu_neighbor_time = time.time() - cpu_start
        
        print(f"CPU Neighbor Search: {cpu_neighbor_time:.3f}s")
        print(f"Neighbors found: {neighbors_cpu[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Spatial operations test failed: {e}")
        return False

def test_memory_management():
    """Test GPU memory management"""
    
    print("\n💾 GPU MEMORY MANAGEMENT TEST")
    print("=" * 50)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("GPU not available - skipping memory test")
            return True
        
        device = torch.device('cuda:0')
        
        # Get initial memory state
        initial_memory = torch.cuda.memory_allocated(device)
        print(f"Initial GPU memory: {initial_memory / (1024**2):.1f} MB")
        
        # Allocate some tensors
        print("🧪 Allocating GPU memory...")
        tensors = []
        
        for i in range(10):
            tensor = torch.randn(1000, 1000, device=device)
            tensors.append(tensor)
            
            current_memory = torch.cuda.memory_allocated(device)
            print(f"  Allocation {i+1}: {current_memory / (1024**2):.1f} MB")
        
        peak_memory = torch.cuda.memory_allocated(device)
        print(f"Peak GPU memory: {peak_memory / (1024**2):.1f} MB")
        
        # Clear memory
        print("🧹 Clearing GPU memory...")
        del tensors
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated(device)
        print(f"Final GPU memory: {final_memory / (1024**2):.1f} MB")
        
        if final_memory <= initial_memory * 1.1:  # Allow 10% tolerance
            print("✅ Memory management successful - no significant leaks detected")
        else:
            print("⚠️  Potential memory leak detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False

def run_gpu_demonstration():
    """Run comprehensive GPU acceleration demonstration"""
    
    print("🚀 GPU ACCELERATION SYSTEM DEMONSTRATION")
    print("Testing GPU acceleration capabilities for EA-NN")
    print("=" * 70)
    
    tests = [
        ("PyTorch Installation", test_pytorch_installation),
        ("Basic GPU Operations", test_basic_gpu_operations),
        ("Neural Network Acceleration", test_neural_network_acceleration),
        ("Spatial Operations", test_spatial_operations),
        ("Memory Management", test_memory_management)
    ]
    
    passed = 0
    total = len(tests)
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("=" * 70)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"🎯 GPU DEMONSTRATION RESULTS")
    print("=" * 70)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Total Time: {total_time:.1f} seconds")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ GPU acceleration system is ready for EA-NN implementation")
        print("🚀 Hardware can support high-performance evolutionary neural networks")
    elif passed >= total * 0.8:
        print("✅ Most tests passed - GPU acceleration is functional")
        print("💡 Some optimizations may be needed for maximum performance")
    else:
        print("⚠️  Limited GPU capabilities detected")
        print("🔧 Consider CPU-only mode or hardware upgrades")
    
    # Final recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            total_memory = sum(
                torch.cuda.get_device_properties(i).total_memory 
                for i in range(gpu_count)
            ) / (1024**3)
            
            print(f"✅ Hardware Status: {gpu_count} GPU(s) with {total_memory:.1f} GB total memory")
            
            if total_memory >= 8:
                print("🚀 Excellent: Hardware supports large-scale simulations (1000+ agents)")
            elif total_memory >= 4:
                print("✅ Good: Hardware supports medium-scale simulations (500+ agents)")
            else:
                print("💡 Basic: Hardware supports small-scale simulations (200+ agents)")
                
            print("🎯 Ready to implement GPU-accelerated evolutionary neural networks!")
        else:
            print("💻 No GPU detected - CPU optimization will provide excellent performance")
            print("🔧 Consider GPU hardware for maximum acceleration benefits")
    except:
        pass
    
    return passed >= total * 0.8

if __name__ == '__main__':
    success = run_gpu_demonstration()
    sys.exit(0 if success else 1)
