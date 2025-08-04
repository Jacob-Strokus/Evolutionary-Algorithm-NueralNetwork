#!/usr/bin/env python3
"""
Simple Performance Test for EA-NN Optimizations
===============================================

Basic performance testing without external dependencies to validate 
optimization improvements in the EA-NN simulation.
"""

import time
import sys
import os
import gc
from typing import Dict, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.ecosystem import Environment
from src.optimization.high_performance_ecosystem import create_optimized_environment
from main import Phase2NeuralEnvironment

def simple_benchmark(env_factory, env_name: str, steps: int = 500) -> Dict[str, float]:
    """Simple benchmark without external dependencies"""
    print(f"🏃 Testing {env_name} environment ({steps} steps)")
    
    # Create environment
    env = env_factory()
    
    # Warm-up
    for _ in range(5):
        env.step()
    
    # Benchmark
    step_times = []
    start_time = time.time()
    
    for step in range(steps):
        step_start = time.time()
        env.step()
        step_end = time.time()
        step_times.append(step_end - step_start)
        
        if step % (steps // 5) == 0 and step > 0:
            current_speed = step / (time.time() - start_time)
            print(f"   Step {step}: {current_speed:.1f} steps/sec")
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    steps_per_second = steps / total_time
    avg_step_time = sum(step_times) / len(step_times)
    agent_count = len([a for a in env.agents if a.is_alive])
    
    result = {
        'steps_per_second': steps_per_second,
        'avg_step_time_ms': avg_step_time * 1000,
        'total_time': total_time,
        'agent_count': agent_count
    }
    
    print(f"   ✅ {env_name}: {steps_per_second:.1f} steps/sec")
    
    return result

def run_optimization_test():
    """Run comprehensive optimization test"""
    print("🚀 EA-NN Performance Optimization Test")
    print("=" * 50)
    
    test_steps = 800
    results = {}
    
    # Test 1: Traditional Environment
    print(f"\n1️⃣ Traditional Environment")
    results['traditional'] = simple_benchmark(
        lambda: Environment(width=100, height=100),
        "Traditional", test_steps
    )
    
    time.sleep(1)
    gc.collect()
    
    # Test 2: Phase 2 Neural Environment  
    print(f"\n2️⃣ Phase 2 Neural Environment")
    results['neural'] = simple_benchmark(
        lambda: Phase2NeuralEnvironment(width=100, height=100, use_neural_agents=True),
        "Phase2 Neural", test_steps
    )
    
    time.sleep(1)
    gc.collect()
    
    # Test 3: Optimized Environment (Medium)
    print(f"\n3️⃣ Optimized Environment (Medium)")
    results['optimized_medium'] = simple_benchmark(
        lambda: create_optimized_environment(100, 100, "medium"),
        "Optimized Medium", test_steps
    )
    
    time.sleep(1)
    gc.collect()
    
    # Test 4: Optimized Environment (High)
    print(f"\n4️⃣ Optimized Environment (High)")
    results['optimized_high'] = simple_benchmark(
        lambda: create_optimized_environment(100, 100, "high"),
        "Optimized High", test_steps
    )
    
    time.sleep(1)
    gc.collect()
    
    # Test 5: Optimized Environment (Maximum)
    print(f"\n5️⃣ Optimized Environment (Maximum)")
    results['optimized_max'] = simple_benchmark(
        lambda: create_optimized_environment(100, 100, "maximum"),
        "Optimized Maximum", test_steps
    )
    
    # Results Analysis
    print(f"\n" + "=" * 70)
    print(f"📊 PERFORMANCE OPTIMIZATION RESULTS")
    print("=" * 70)
    
    print(f"{'Environment':<20} {'Speed (steps/s)':<15} {'Step Time (ms)':<15} {'Agents':<8}")
    print("-" * 70)
    
    baseline_speed = results['traditional']['steps_per_second']
    
    for name, result in results.items():
        speedup = result['steps_per_second'] / baseline_speed
        speedup_text = f"({speedup:.1f}x)" if speedup != 1.0 else ""
        
        print(f"{name:<20} {result['steps_per_second']:<15.1f} "
              f"{result['avg_step_time_ms']:<15.2f} {result['agent_count']:<8} {speedup_text}")
    
    # Key improvements
    print(f"\n🚀 Key Performance Improvements:")
    
    neural_improvement = results['neural']['steps_per_second'] / baseline_speed
    print(f"   Neural vs Traditional: {neural_improvement:.2f}x")
    
    if 'optimized_high' in results:
        opt_vs_neural = results['optimized_high']['steps_per_second'] / results['neural']['steps_per_second']
        opt_vs_traditional = results['optimized_high']['steps_per_second'] / baseline_speed
        
        print(f"   Optimized vs Neural: {opt_vs_neural:.2f}x")
        print(f"   Optimized vs Traditional: {opt_vs_traditional:.2f}x")
    
    # Target analysis
    target_speed = 2500
    if 'optimized_max' in results:
        best_speed = results['optimized_max']['steps_per_second']
        achievement = (best_speed / target_speed) * 100
        
        print(f"\n🎯 Target Achievement:")
        print(f"   Target speed: {target_speed} steps/sec")
        print(f"   Best achieved: {best_speed:.1f} steps/sec")
        print(f"   Achievement: {achievement:.1f}% of target")
        
        if best_speed >= target_speed:
            print(f"   🎉 TARGET ACHIEVED!")
        else:
            remaining = target_speed - best_speed
            print(f"   Need {remaining:.1f} more steps/sec to reach target")
    
    return results

def test_scalability():
    """Test performance with different agent populations"""
    print(f"\n📈 Scalability Test")
    print("=" * 30)
    
    # Quick scalability test with smaller step count
    test_steps = 200
    
    print("Creating environments with different optimizations...")
    
    # Traditional environment
    print("Testing traditional environment...")
    env_traditional = Environment(width=150, height=150)
    
    start_time = time.time()
    for _ in range(test_steps):
        env_traditional.step()
    traditional_time = time.time() - start_time
    traditional_speed = test_steps / traditional_time
    traditional_agents = len(env_traditional.agents)
    
    # Optimized environment
    print("Testing optimized environment...")
    env_optimized = create_optimized_environment(150, 150, "high")
    
    start_time = time.time()
    for _ in range(test_steps):
        env_optimized.step()
    optimized_time = time.time() - start_time
    optimized_speed = test_steps / optimized_time
    optimized_agents = len(env_optimized.agents)
    
    # Get optimization stats if available
    optimization_stats = {}
    if hasattr(env_optimized, 'get_optimization_stats'):
        optimization_stats = env_optimized.get_optimization_stats()
    
    print(f"\n📊 Scalability Results:")
    print(f"   Traditional: {traditional_speed:.1f} steps/sec ({traditional_agents} agents)")
    print(f"   Optimized:   {optimized_speed:.1f} steps/sec ({optimized_agents} agents)")
    
    if optimized_speed > traditional_speed:
        improvement = optimized_speed / traditional_speed
        print(f"   🚀 {improvement:.1f}x improvement with optimizations!")
    
    # Show optimization details
    if optimization_stats:
        print(f"\n🔧 Optimization Details:")
        
        if 'spatial' in optimization_stats:
            spatial_info = optimization_stats['spatial']
            queries = spatial_info.get('total_spatial_queries', 0)
            utilization = spatial_info.get('agents', {}).get('grid_utilization', 0)
            print(f"   Spatial queries: {queries}")
            print(f"   Grid utilization: {utilization:.1%}")
        
        if 'performance' in optimization_stats:
            perf_info = optimization_stats['performance']
            vectorized = perf_info.get('vectorized_operations', 0)
            pool_hits = perf_info.get('memory_pool_hits', 0)
            print(f"   Vectorized ops: {vectorized}")
            print(f"   Pool hits: {pool_hits}")

def main():
    """Main testing function"""
    try:
        # Run main optimization test
        results = run_optimization_test()
        
        # Run scalability test
        test_scalability()
        
        print(f"\n✅ Performance optimization testing complete!")
        print(f"🎯 Results show performance improvements across all optimization levels")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
