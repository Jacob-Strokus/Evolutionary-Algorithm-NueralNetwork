#!/usr/bin/env python3
"""
Quick Performance Validation
============================

Quick test to validate that our performance optimizations are working correctly.
"""

import time
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.ecosystem import Environment
from src.optimization.high_performance_ecosystem import create_optimized_environment

def quick_test():
    """Quick performance test"""
    print("🚀 Quick Performance Validation")
    print("=" * 40)
    
    test_steps = 100
    
    # Test traditional environment
    print("Testing traditional environment...")
    env1 = Environment(width=80, height=80)
    
    start_time = time.time()
    for _ in range(test_steps):
        env1.step()
    traditional_time = time.time() - start_time
    traditional_speed = test_steps / traditional_time
    
    print(f"Traditional: {traditional_speed:.1f} steps/sec")
    
    # Test optimized environment
    print("Testing optimized environment...")
    env2 = create_optimized_environment(80, 80, "high")
    
    start_time = time.time()
    for _ in range(test_steps):
        env2.step()
    optimized_time = time.time() - start_time
    optimized_speed = test_steps / optimized_time
    
    print(f"Optimized:   {optimized_speed:.1f} steps/sec")
    
    # Compare
    if optimized_speed > traditional_speed:
        improvement = optimized_speed / traditional_speed
        print(f"🎉 {improvement:.1f}x improvement with optimizations!")
    else:
        print(f"⚠️ Optimizations may need tuning")
    
    # Show optimization stats
    if hasattr(env2, 'get_optimization_stats'):
        stats = env2.get_optimization_stats()
        print(f"\nOptimization stats:")
        if 'spatial' in stats:
            spatial_queries = stats['spatial'].get('total_spatial_queries', 0)
            print(f"  Spatial queries: {spatial_queries}")
        if 'performance' in stats:
            perf = stats['performance']
            print(f"  Vectorized ops: {perf.get('vectorized_operations', 0)}")
            print(f"  Pool hits: {perf.get('memory_pool_hits', 0)}")

if __name__ == "__main__":
    quick_test()
