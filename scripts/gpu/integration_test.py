#!/usr/bin/env python3
"""
GPU Acceleration Integration Test
===============================

Integration test showing how GPU acceleration works with the existing EA-NN system.
Demonstrates seamless CPU fallback and performance benefits.
"""

import sys
import os
import time
import numpy as np

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

def test_gpu_integration():
    """Test GPU acceleration integration"""
    
    print("🔗 GPU ACCELERATION INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Import our GPU acceleration system
        from optimization.gpu_accelerated_ecosystem_practical import (
            GPUAcceleratedEcosystemPractical,
            GPUAcceleratedNeuralNetwork,
            GPUAcceleratedSpatialOperations,
            benchmark_gpu_performance
        )
        
        print("✅ GPU acceleration modules imported successfully")
        
        # Test 1: Neural Network Acceleration
        print("\n🧠 Testing Neural Network Acceleration...")
        
        neural_net = GPUAcceleratedNeuralNetwork(
            input_size=10, 
            hidden_size=15, 
            output_size=4,
            use_gpu=True
        )
        
        # Test single processing
        test_input = np.random.randn(10)
        single_output = neural_net.forward_single(test_input)
        print(f"✅ Single neural processing: output shape {single_output.shape}")
        
        # Test batch processing
        batch_input = np.random.randn(100, 10)
        batch_output = neural_net.forward_batch(batch_input)
        print(f"✅ Batch neural processing: {batch_input.shape} -> {batch_output.shape}")
        
        # Test 2: Spatial Operations
        print("\n🌍 Testing Spatial Operations...")
        
        spatial_ops = GPUAcceleratedSpatialOperations(use_gpu=True)
        
        # Test distance matrix
        positions = np.random.randn(100, 2) * 100
        distance_matrix = spatial_ops.calculate_distance_matrix(positions)
        print(f"✅ Distance matrix: {positions.shape} -> {distance_matrix.shape}")
        
        # Test neighbor finding
        query_positions = positions[:10]
        neighbors = spatial_ops.find_neighbors_within_radius(
            positions, query_positions, radius=50.0
        )
        print(f"✅ Neighbor finding: {len(neighbors)} queries processed")
        
        # Test 3: Full Ecosystem Integration
        print("\n🚀 Testing Full Ecosystem Integration...")
        
        ecosystem = GPUAcceleratedEcosystemPractical(
            width=800,
            height=600,
            use_gpu=True,
            gpu_threshold=50
        )
        
        # Add test agents
        for i in range(150):
            agent = {
                'id': i,
                'x': np.random.uniform(0, 800),
                'y': np.random.uniform(0, 600),
                'energy': np.random.uniform(0.5, 1.0),
                'age': np.random.randint(0, 100),
                'vision_range': np.random.uniform(30, 70),
                'speed': np.random.uniform(0.5, 2.0),
                'neighbors': [],
                'last_food_time': np.random.randint(0, 50),
                'reproduction_readiness': np.random.uniform(0, 1)
            }
            ecosystem.add_agent(agent)
        
        print(f"✅ Ecosystem created with {len(ecosystem.agents)} agents")
        
        # Test ecosystem steps
        step_times = []
        for i in range(10):
            result = ecosystem.step_optimized()
            step_times.append(result['step_time'])
            
            if i == 0:
                print(f"✅ First step: {result['step_time']:.3f}s, Mode: {result['mode']}")
        
        avg_step_time = np.mean(step_times)
        processing_rate = len(ecosystem.agents) / avg_step_time
        
        print(f"✅ Average step time: {avg_step_time:.3f}s")
        print(f"✅ Processing rate: {processing_rate:.1f} agents/sec")
        
        # Performance summary
        summary = ecosystem.get_performance_summary()
        print(f"\n📊 Performance Summary:")
        print(f"   GPU Available: {summary['gpu_available']}")
        print(f"   Processing Mode: {summary['processing_mode']}")
        print(f"   Total Speedup: {summary['total_speedup']:.2f}x")
        print(f"   Neural Network Time: {summary['neural_network_time']:.3f}s")
        print(f"   Spatial Operations Time: {summary['spatial_operations_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare CPU vs GPU acceleration performance"""
    
    print("\n⚡ PERFORMANCE COMPARISON TEST")
    print("=" * 60)
    
    try:
        from optimization.gpu_accelerated_ecosystem_practical import GPUAcceleratedNeuralNetwork
        
        # Test different batch sizes
        batch_sizes = [10, 50, 100, 200]
        
        for batch_size in batch_sizes:
            print(f"\n🧪 Testing batch size: {batch_size}")
            
            # Create neural network
            neural_net = GPUAcceleratedNeuralNetwork(
                input_size=10,
                hidden_size=15, 
                output_size=4,
                use_gpu=True
            )
            
            # Generate test data
            test_inputs = np.random.randn(batch_size, 10)
            
            # Batch processing time
            start_time = time.time()
            batch_outputs = neural_net.forward_batch(test_inputs)
            batch_time = time.time() - start_time
            
            # Single processing time (for comparison)
            start_time = time.time()
            single_outputs = []
            for i in range(batch_size):
                output = neural_net.forward_single(test_inputs[i])
                single_outputs.append(output)
            single_time = time.time() - start_time
            
            # Calculate performance metrics
            batch_rate = batch_size / batch_time
            single_rate = batch_size / single_time
            speedup = single_time / batch_time
            
            print(f"   Batch processing: {batch_time:.3f}s ({batch_rate:.1f} agents/sec)")
            print(f"   Single processing: {single_time:.3f}s ({single_rate:.1f} agents/sec)")
            print(f"   Batch speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False

def main():
    """Run comprehensive GPU acceleration integration tests"""
    
    print("🚀 GPU ACCELERATION INTEGRATION TESTING")
    print("Testing integration with existing EA-NN ecosystem")
    print("=" * 80)
    
    tests = [
        ("GPU Integration Test", test_gpu_integration),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    passed = 0
    total = len(tests)
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("=" * 80)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"🎯 INTEGRATION TEST RESULTS")
    print("=" * 80)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Total Time: {total_time:.1f} seconds")
    
    if passed == total:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ GPU acceleration system is fully integrated and operational")
        print("🚀 Ready for high-performance evolutionary neural network research!")
    else:
        print("⚠️  Some integration tests failed")
        print("🔧 Review the output above for specific issues")
    
    # Final status
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n🎯 SYSTEM STATUS: GPU-accelerated mode ready")
            print(f"   PyTorch: {torch.__version__}")
            print(f"   CUDA: Available")
        else:
            print(f"\n🎯 SYSTEM STATUS: CPU-optimized mode ready")
            print(f"   PyTorch: {torch.__version__} (CPU)")
            print(f"   Performance: Excellent with seamless fallback")
    except:
        print(f"\n🎯 SYSTEM STATUS: NumPy fallback mode")
        print(f"   GPU acceleration will use CPU fallback")
    
    return passed >= total * 0.8

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
