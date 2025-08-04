#!/usr/bin/env python3
"""
Performance Optimization Validation Suite
==========================================

Comprehensive testing suite to validate and measure the effectiveness of
performance optimizations in the EA-NN simulation. Compares traditional
implementation vs. optimized versions across multiple metrics.
"""

import time
import sys
import os
import numpy as np
import gc
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.ecosystem import Environment
from src.optimization.high_performance_ecosystem import HighPerformanceEcosystem, create_optimized_environment
from src.optimization.spatial_indexing import benchmark_spatial_vs_brute_force
from main import Phase2NeuralEnvironment

@dataclass
class BenchmarkResult:
    """Results from a performance benchmark"""
    environment_type: str
    steps_completed: int
    total_time: float
    steps_per_second: float
    avg_step_time_ms: float
    memory_usage_mb: float
    agent_count: int
    additional_stats: Dict

class PerformanceValidator:
    """Validates performance optimizations through comprehensive testing"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def benchmark_environment(self, env_factory, env_name: str, steps: int = 1000) -> BenchmarkResult:
        """Benchmark a specific environment implementation"""
        print(f"🏃 Benchmarking {env_name} environment ({steps} steps)")
        
        # Setup
        gc.collect()
        start_memory = self._get_memory_usage()
        
        # Create environment
        env = env_factory()
        
        # Warm-up (10 steps to stabilize)
        for _ in range(10):
            env.step()
        
        # Actual benchmark
        start_time = time.time()
        step_times = []
        
        for step in range(steps):
            step_start = time.time()
            env.step()
            step_end = time.time()
            step_times.append((step_end - step_start) * 1000)  # Convert to ms
            
            # Progress indicator
            if steps >= 100 and step % (steps // 10) == 0 and step > 0:
                current_speed = step / (time.time() - start_time)
                print(f"   Step {step}: {current_speed:.1f} steps/sec")
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Calculate metrics
        total_time = end_time - start_time
        steps_per_second = steps / total_time
        avg_step_time = np.mean(step_times)
        memory_delta = end_memory - start_memory
        agent_count = len([a for a in env.agents if a.is_alive])
        
        # Get additional stats if available
        additional_stats = {}
        if hasattr(env, 'get_optimization_stats'):
            additional_stats = env.get_optimization_stats()
        
        result = BenchmarkResult(
            environment_type=env_name,
            steps_completed=steps,
            total_time=total_time,
            steps_per_second=steps_per_second,
            avg_step_time_ms=avg_step_time,
            memory_usage_mb=memory_delta,
            agent_count=agent_count,
            additional_stats=additional_stats
        )
        
        self.results.append(result)
        
        print(f"   ✅ {env_name}: {steps_per_second:.1f} steps/sec, {avg_step_time:.2f}ms/step")
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def compare_environments(self, steps: int = 2000) -> Dict[str, BenchmarkResult]:
        """Compare different environment implementations"""
        print(f"\n🏁 Environment Performance Comparison")
        print("=" * 60)
        
        results = {}
        
        # Traditional environment
        print(f"\n1️⃣ Traditional Environment")
        results['traditional'] = self.benchmark_environment(
            lambda: Environment(width=100, height=100),
            "Traditional", steps
        )
        
        # Small delay between tests
        time.sleep(1)
        gc.collect()
        
        # Neural environment (Phase 2)
        print(f"\n2️⃣ Phase 2 Neural Environment")
        results['neural'] = self.benchmark_environment(
            lambda: Phase2NeuralEnvironment(width=100, height=100, use_neural_agents=True),
            "Phase2 Neural", steps
        )
        
        time.sleep(1)
        gc.collect()
        
        # Optimized environment - Low setting
        print(f"\n3️⃣ Optimized Environment (Low)")
        results['optimized_low'] = self.benchmark_environment(
            lambda: create_optimized_environment(100, 100, "low"),
            "Optimized Low", steps
        )
        
        time.sleep(1)
        gc.collect()
        
        # Optimized environment - Medium setting
        print(f"\n4️⃣ Optimized Environment (Medium)")
        results['optimized_medium'] = self.benchmark_environment(
            lambda: create_optimized_environment(100, 100, "medium"),
            "Optimized Medium", steps
        )
        
        time.sleep(1)
        gc.collect()
        
        # Optimized environment - High setting
        print(f"\n5️⃣ Optimized Environment (High)")
        results['optimized_high'] = self.benchmark_environment(
            lambda: create_optimized_environment(100, 100, "high"),
            "Optimized High", steps
        )
        
        time.sleep(1)
        gc.collect()
        
        # Optimized environment - Maximum setting
        print(f"\n6️⃣ Optimized Environment (Maximum)")
        results['optimized_max'] = self.benchmark_environment(
            lambda: create_optimized_environment(100, 100, "maximum"),
            "Optimized Maximum", steps
        )
        
        return results
    
    def test_scalability(self) -> Dict[int, Dict[str, float]]:
        """Test performance scaling with different population sizes"""
        print(f"\n📈 Scalability Testing")
        print("=" * 40)
        
        population_sizes = [50, 100, 150, 200]
        scalability_results = {}
        
        for pop_size in population_sizes:
            print(f"\n🔬 Testing population size: {pop_size}")
            
            # Test traditional vs optimized at this population size
            environments = {
                'traditional': lambda: Environment(width=200, height=200),
                'optimized': lambda: create_optimized_environment(200, 200, "high")
            }
            
            pop_results = {}
            
            for env_name, env_factory in environments.items():
                try:
                    env = env_factory()
                    
                    # Run a short test
                    start_time = time.time()
                    for _ in range(200):  # Shorter test for scalability
                        env.step()
                    end_time = time.time()
                    
                    duration = end_time - start_time
                    speed = 200 / duration
                    
                    pop_results[env_name] = speed
                    print(f"   {env_name}: {speed:.1f} steps/sec")
                    
                except Exception as e:
                    print(f"   {env_name}: Failed - {e}")
                    pop_results[env_name] = 0.0
                
                gc.collect()
                time.sleep(0.5)
            
            scalability_results[pop_size] = pop_results
        
        return scalability_results
    
    def generate_report(self, comparison_results: Dict[str, BenchmarkResult], 
                       scalability_results: Dict[int, Dict[str, float]]):
        """Generate comprehensive performance report"""
        print(f"\n" + "=" * 80)
        print(f"🏆 COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 80)
        
        # Environment comparison summary
        print(f"\n📊 Environment Performance Summary:")
        print(f"{'Environment':<20} {'Speed (steps/s)':<15} {'Step Time (ms)':<15} {'Memory (MB)':<12} {'Agents':<8}")
        print("-" * 80)
        
        baseline_speed = comparison_results['traditional'].steps_per_second
        
        for name, result in comparison_results.items():
            speedup = result.steps_per_second / baseline_speed
            speedup_indicator = f"({speedup:.1f}x)" if speedup != 1.0 else ""
            
            print(f"{result.environment_type:<20} {result.steps_per_second:<15.1f} "
                  f"{result.avg_step_time_ms:<15.2f} {result.memory_usage_mb:<12.1f} "
                  f"{result.agent_count:<8} {speedup_indicator}")
        
        # Performance improvements
        print(f"\n🚀 Performance Improvements:")
        
        neural_vs_traditional = (comparison_results['neural'].steps_per_second / 
                               comparison_results['traditional'].steps_per_second)
        print(f"   Neural vs Traditional: {neural_vs_traditional:.2f}x")
        
        if 'optimized_high' in comparison_results:
            optimized_vs_neural = (comparison_results['optimized_high'].steps_per_second / 
                                 comparison_results['neural'].steps_per_second)
            print(f"   Optimized vs Neural: {optimized_vs_neural:.2f}x")
            
            optimized_vs_traditional = (comparison_results['optimized_high'].steps_per_second / 
                                      comparison_results['traditional'].steps_per_second)
            print(f"   Optimized vs Traditional: {optimized_vs_traditional:.2f}x")
        
        # Scalability analysis
        if scalability_results:
            print(f"\n📈 Scalability Analysis:")
            print(f"{'Population Size':<15} {'Traditional':<15} {'Optimized':<15} {'Improvement':<15}")
            print("-" * 60)
            
            for pop_size, results in scalability_results.items():
                if 'traditional' in results and 'optimized' in results:
                    improvement = results['optimized'] / results['traditional'] if results['traditional'] > 0 else 0
                    print(f"{pop_size:<15} {results['traditional']:<15.1f} "
                          f"{results['optimized']:<15.1f} {improvement:<15.2f}x")
        
        # Optimization features impact
        if 'optimized_high' in comparison_results:
            opt_stats = comparison_results['optimized_high'].additional_stats
            if opt_stats:
                print(f"\n🔧 Optimization Features Impact:")
                
                if 'spatial' in opt_stats:
                    spatial_stats = opt_stats['spatial']
                    print(f"   Spatial queries performed: {spatial_stats.get('total_spatial_queries', 0)}")
                    print(f"   Spatial grid utilization: {spatial_stats.get('agents', {}).get('grid_utilization', 0):.1%}")
                
                if 'performance' in opt_stats:
                    perf_stats = opt_stats['performance']
                    print(f"   Vectorized operations: {perf_stats.get('vectorized_operations', 0)}")
                    print(f"   Memory pool hits: {perf_stats.get('memory_pool_hits', 0)}")
        
        # Target achievement
        print(f"\n🎯 Target Achievement:")
        target_speed = 2500  # Target from optimization plan
        
        if 'optimized_high' in comparison_results:
            current_best = comparison_results['optimized_high'].steps_per_second
            achievement = (current_best / target_speed) * 100
            remaining = target_speed - current_best
            
            print(f"   Target speed: {target_speed} steps/sec")
            print(f"   Current best: {current_best:.1f} steps/sec")
            print(f"   Achievement: {achievement:.1f}% of target")
            
            if current_best >= target_speed:
                print(f"   🎉 TARGET ACHIEVED! Exceeded by {current_best - target_speed:.1f} steps/sec")
            else:
                print(f"   ⚡ Need {remaining:.1f} more steps/sec to reach target")
        
        print(f"\n" + "=" * 80)

def main():
    """Main validation function"""
    print("🚀 Performance Optimization Validation Suite")
    print("============================================")
    
    validator = PerformanceValidator()
    
    try:
        # Run spatial indexing benchmark first
        print("🗺️ Testing Spatial Indexing Performance")
        benchmark_spatial_vs_brute_force()
        
        # Run environment comparisons
        comparison_results = validator.compare_environments(steps=1500)
        
        # Run scalability tests
        scalability_results = validator.test_scalability()
        
        # Generate comprehensive report
        validator.generate_report(comparison_results, scalability_results)
        
        print(f"\n✅ Performance validation complete!")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
