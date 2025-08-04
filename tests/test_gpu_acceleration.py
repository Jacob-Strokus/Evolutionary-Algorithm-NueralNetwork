#!/usr/bin/env python3
"""
GPU Acceleration Testing and Demonstration
==========================================

Comprehensive testing suite for GPU-accelerated EA-NN components.
Tests hardware detection, neural network acceleration, spatial operations,
and complete ecosystem performance with detailed benchmarking.
"""

import sys
import os
import time
import numpy as np
import logging
from typing import Dict, List, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_gpu_hardware_detection():
    """Test GPU hardware detection and capabilities"""
    
    print("🔍 GPU HARDWARE DETECTION TEST")
    print("=" * 50)
    
    try:
        from optimization.gpu_manager import initialize_gpu_acceleration, GPUConfig
        
        # Initialize GPU system
        config = GPUConfig(
            memory_fraction=0.8,
            enable_mixed_precision=True,
            fallback_enabled=True
        )
        
        gpu_manager = initialize_gpu_acceleration(config)
        
        print("\n✅ GPU hardware detection completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ GPU hardware detection failed: {e}")
        return False

def test_gpu_neural_networks():
    """Test GPU-accelerated neural networks"""
    
    print("\n🧠 GPU NEURAL NETWORK TEST")
    print("=" * 50)
    
    try:
        from optimization.gpu_neural_networks import create_gpu_neural_processor, NeuralNetworkConfig
        
        # Create neural processor
        config = NeuralNetworkConfig(
            input_size=10,
            hidden_sizes=[20, 15],
            output_size=4,
            mixed_precision=True
        )
        
        processor = create_gpu_neural_processor(config)
        processor.print_status()
        
        # Test single agent processing
        test_state = np.random.randn(10).astype(np.float32)
        output = processor.process_agent(test_state)
        
        print(f"\n🧪 Single Agent Test:")
        print(f"   Input shape: {test_state.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Test batch processing
        batch_states = [np.random.randn(10).astype(np.float32) for _ in range(32)]
        batch_outputs = processor.process_agent_batch(batch_states)
        
        print(f"\n🔄 Batch Processing Test:")
        print(f"   Batch size: {len(batch_states)}")
        print(f"   Output count: {len(batch_outputs)}")
        print(f"   Output shapes: {[out.shape for out in batch_outputs[:3]]}...")
        
        # Performance benchmark
        print(f"\n⚡ Performance Benchmark:")
        benchmark_results = processor.benchmark_performance([10, 50, 100])
        
        for agent_count, results in benchmark_results.items():
            print(f"   {agent_count}: {results['agents_per_second']:.1f} agents/sec ({results['processor_type']})")
        
        print("✅ Neural network test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_spatial_operations():
    """Test GPU-accelerated spatial operations"""
    
    print("\n🌍 GPU SPATIAL OPERATIONS TEST")
    print("=" * 50)
    
    try:
        from optimization.gpu_spatial_operations import create_gpu_spatial_processor, SpatialConfig
        
        # Create spatial processor
        config = SpatialConfig(
            use_gpu=True,
            batch_size=1000,
            distance_threshold=100.0
        )
        
        processor = create_gpu_spatial_processor(config)
        processor.print_status()
        
        # Test distance matrix calculation
        positions = np.random.randn(100, 2).astype(np.float32) * 50
        
        print(f"\n📏 Distance Matrix Test:")
        start_time = time.time()
        distance_matrix = processor.calculate_distance_matrix_gpu(positions)
        elapsed = time.time() - start_time
        
        print(f"   Positions: {positions.shape}")
        print(f"   Distance matrix: {distance_matrix.shape}")
        print(f"   Calculation time: {elapsed:.3f}s")
        print(f"   Distance range: [{distance_matrix.min():.1f}, {distance_matrix.max():.1f}]")
        
        # Test neighbor finding
        query_positions = positions[:10]  # First 10 as queries
        radius = 25.0
        
        print(f"\n🔍 Neighbor Search Test:")
        start_time = time.time()
        neighbors = processor.find_neighbors_gpu(query_positions, positions, radius)
        elapsed = time.time() - start_time
        
        print(f"   Query positions: {len(query_positions)}")
        print(f"   Search radius: {radius}")
        print(f"   Search time: {elapsed:.3f}s")
        print(f"   Neighbors found: {[len(n) for n in neighbors[:5]]}...")
        
        # Test position updates
        velocities = np.random.randn(100, 2).astype(np.float32) * 2
        bounds = (0, 800, 0, 600)
        
        print(f"\n🏃 Position Update Test:")
        start_time = time.time()
        new_positions = processor.update_agent_positions_gpu(positions, velocities, bounds)
        elapsed = time.time() - start_time
        
        print(f"   Original positions: {positions.shape}")
        print(f"   New positions: {new_positions.shape}")
        print(f"   Update time: {elapsed:.3f}s")
        print(f"   Position bounds: X[{new_positions[:, 0].min():.1f}, {new_positions[:, 0].max():.1f}], Y[{new_positions[:, 1].min():.1f}, {new_positions[:, 1].max():.1f}]")
        
        # Performance benchmark
        print(f"\n⚡ Spatial Performance Benchmark:")
        benchmark_results = processor.benchmark_spatial_operations([50, 100, 200])
        
        for agent_count, results in benchmark_results.items():
            print(f"   {agent_count}:")
            print(f"     Distance matrix: {results['distance_matrix_time']:.3f}s")
            print(f"     Neighbor search: {results['neighbor_search_time']:.3f}s")
            print(f"     Position update: {results['position_update_time']:.3f}s")
            print(f"     Framework: {results['framework']}")
        
        print("✅ Spatial operations test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Spatial operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_accelerated_ecosystem():
    """Test complete GPU-accelerated ecosystem"""
    
    print("\n⚡ GPU ACCELERATED ECOSYSTEM TEST")
    print("=" * 50)
    
    try:
        from optimization.gpu_accelerated_ecosystem import create_gpu_accelerated_ecosystem, GPUEcosystemConfig
        from core.ecosystem import Agent, SpeciesType, Position
        
        # Create ecosystem configuration
        config = GPUEcosystemConfig(
            enable_gpu=True,
            neural_gpu_threshold=30,
            spatial_gpu_threshold=50,
            adaptive_processing=True,
            performance_monitoring=True
        )
        
        # Create ecosystem
        ecosystem = create_gpu_accelerated_ecosystem(800, 600, config)
        ecosystem.print_status()
        
        # Add test agents
        print(f"\n🧪 Adding Test Agents:")
        agent_id = 0
        
        for i in range(80):
            x = np.random.randint(50, 750)
            y = np.random.randint(50, 550)
            position = Position(x, y)
            
            species = SpeciesType.HERBIVORE if i % 4 != 0 else SpeciesType.CARNIVORE
            agent = Agent(species, position, agent_id)
            
            ecosystem.add_agent(agent)
            agent_id += 1
        
        print(f"   Added {len(ecosystem.agents)} agents")
        print(f"   Herbivores: {sum(1 for a in ecosystem.agents if a.species_type == SpeciesType.HERBIVORE)}")
        print(f"   Carnivores: {sum(1 for a in ecosystem.agents if a.species_type == SpeciesType.CARNIVORE)}")
        
        # Run simulation steps
        print(f"\n🏃 Running Simulation Steps:")
        steps = 50
        
        start_time = time.time()
        
        for step in range(steps):
            ecosystem.step()
            
            if (step + 1) % 10 == 0:
                print(f"   Step {step + 1}/{steps} - {len(ecosystem.agents)} agents alive")
        
        total_time = time.time() - start_time
        steps_per_sec = steps / total_time
        
        print(f"\n📊 Simulation Results:")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Steps per second: {steps_per_sec:.1f}")
        print(f"   Average step time: {total_time/steps:.3f}s")
        print(f"   Final agent count: {len(ecosystem.agents)}")
        
        # Print final status
        print(f"\n📈 Final System Status:")
        ecosystem.print_status()
        
        print("✅ GPU accelerated ecosystem test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ GPU accelerated ecosystem test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare GPU vs CPU performance across different workloads"""
    
    print("\n🏁 GPU vs CPU PERFORMANCE COMPARISON")
    print("=" * 50)
    
    try:
        from optimization.gpu_accelerated_ecosystem import create_gpu_accelerated_ecosystem, GPUEcosystemConfig
        from optimization.high_performance_ecosystem import create_optimized_environment
        from core.ecosystem import Agent, SpeciesType, Position
        
        # Test different population sizes
        population_sizes = [50, 100, 200]
        results = {}
        
        for pop_size in population_sizes:
            print(f"\n🧪 Testing with {pop_size} agents:")
            
            # GPU-accelerated ecosystem
            gpu_config = GPUEcosystemConfig(
                enable_gpu=True,
                neural_gpu_threshold=25,
                spatial_gpu_threshold=50
            )
            gpu_ecosystem = create_gpu_accelerated_ecosystem(800, 600, gpu_config)
            
            # CPU-optimized ecosystem
            cpu_ecosystem = create_optimized_environment(800, 600, "high")
            
            # Add identical agents to both
            for i in range(pop_size):
                x = np.random.randint(50, 750)
                y = np.random.randint(50, 550)
                position = Position(x, y)
                species = SpeciesType.HERBIVORE if i % 3 != 0 else SpeciesType.CARNIVORE
                
                gpu_agent = Agent(species, position, i)
                cpu_agent = Agent(species, Position(x, y), i)
                
                gpu_ecosystem.add_agent(gpu_agent)
                cpu_ecosystem.add_agent(cpu_agent)
            
            # Benchmark GPU ecosystem
            gpu_start = time.time()
            for _ in range(30):
                gpu_ecosystem.step()
            gpu_time = time.time() - gpu_start
            
            # Benchmark CPU ecosystem
            cpu_start = time.time()
            for _ in range(30):
                cpu_ecosystem.step()
            cpu_time = time.time() - cpu_start
            
            # Calculate results
            gpu_steps_per_sec = 30 / gpu_time
            cpu_steps_per_sec = 30 / cpu_time
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
            
            results[pop_size] = {
                'gpu_time': gpu_time,
                'cpu_time': cpu_time,
                'gpu_steps_per_sec': gpu_steps_per_sec,
                'cpu_steps_per_sec': cpu_steps_per_sec,
                'speedup': speedup
            }
            
            print(f"   GPU: {gpu_steps_per_sec:.1f} steps/sec ({gpu_time:.3f}s)")
            print(f"   CPU: {cpu_steps_per_sec:.1f} steps/sec ({cpu_time:.3f}s)")
            
            if speedup > 1:
                print(f"   🚀 GPU {speedup:.2f}x faster")
            elif speedup < 0.8:
                print(f"   ⚠️  GPU {1/speedup:.2f}x slower (overhead)")
            else:
                print(f"   ✅ Similar performance ({speedup:.2f}x)")
        
        # Summary
        print(f"\n📊 PERFORMANCE COMPARISON SUMMARY")
        print("-" * 50)
        print(f"{'Population':<12} {'GPU (steps/s)':<15} {'CPU (steps/s)':<15} {'Speedup'}")
        print("-" * 60)
        
        for pop_size, result in results.items():
            speedup_str = f"{result['speedup']:.2f}x"
            print(f"{pop_size:<12} {result['gpu_steps_per_sec']:<15.1f} {result['cpu_steps_per_sec']:<15.1f} {speedup_str}")
        
        # Find optimal threshold
        optimal_threshold = None
        for pop_size, result in results.items():
            if result['speedup'] > 1.1:  # 10% improvement threshold
                optimal_threshold = pop_size
                break
        
        if optimal_threshold:
            print(f"\n🎯 GPU becomes beneficial at ~{optimal_threshold} agents")
        else:
            print(f"\n💡 GPU acceleration shows overhead at tested scales")
            print(f"   Try larger populations (500+) for maximum GPU benefits")
        
        print("✅ Performance comparison completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_gpu_acceleration_tests():
    """Run comprehensive GPU acceleration test suite"""
    
    print("🚀 GPU ACCELERATION TEST SUITE")
    print("Testing all GPU acceleration components")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    tests = [
        ("Hardware Detection", test_gpu_hardware_detection),
        ("Neural Networks", test_gpu_neural_networks),
        ("Spatial Operations", test_gpu_spatial_operations),
        ("Accelerated Ecosystem", test_gpu_accelerated_ecosystem),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    passed = 0
    total = len(tests)
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running Test: {test_name}")
        print("=" * 60)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"🎯 GPU ACCELERATION TEST RESULTS")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Total Time: {total_time:.1f} seconds")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! GPU acceleration system is working correctly.")
        print("\n🚀 Ready for high-performance evolutionary neural network research!")
    elif passed >= total * 0.8:
        print("✅ Most tests passed - GPU acceleration is functional with some limitations.")
        print("🔧 Review failed tests for optimization opportunities.")
    else:
        print("⚠️  Multiple test failures detected.")
        print("🔧 GPU acceleration may need hardware or configuration adjustments.")
    
    return passed == total

if __name__ == '__main__':
    success = run_gpu_acceleration_tests()
    sys.exit(0 if success else 1)
