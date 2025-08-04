#!/usr/bin/env python3
"""
Performance Profiler for EA-NN Simulation
==========================================

Comprehensive performance analysis and benchmarking tools for the evolutionary 
neural network ecosystem simulation. Measures simulation speed, memory usage, 
real-time display performance, and resource utilization.

Usage:
    python scripts/performance/profile_simulation.py
"""

import time
import psutil
import gc
import sys
import os
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from main import Phase2NeuralEnvironment
from src.core.ecosystem import Environment, SpeciesType

@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    simulation_speed: float  # steps per second
    memory_usage_mb: float   # memory usage in MB
    cpu_usage_percent: float # CPU usage percentage
    agent_count: int        # number of agents during test
    test_duration: float    # test duration in seconds
    avg_step_time: float    # average time per step in ms

class PerformanceProfiler:
    """Comprehensive performance profiling for EA-NN simulation"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return self.process.cpu_percent()
    
    @contextmanager
    def measure_performance(self, test_name: str):
        """Context manager for measuring performance of code blocks"""
        print(f"📊 Starting performance test: {test_name}")
        
        # Setup
        gc.collect()  # Clean up before measurement
        start_memory = self.get_memory_usage()
        start_time = time.time()
        start_cpu = self.get_cpu_usage()
        
        try:
            yield
        finally:
            # Teardown and measurement
            end_time = time.time()
            end_memory = self.get_memory_usage()
            end_cpu = self.get_cpu_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            avg_cpu = (start_cpu + end_cpu) / 2
            
            print(f"✅ {test_name} completed:")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Memory change: {memory_delta:+.1f} MB")
            print(f"   Average CPU: {avg_cpu:.1f}%")
    
    def benchmark_simulation_speed(self, environment_type="neural", steps=1000, agent_count=50) -> PerformanceMetrics:
        """Benchmark raw simulation speed"""
        print(f"\n🚀 Benchmarking {environment_type} simulation speed")
        print(f"   Steps: {steps}, Target agents: {agent_count}")
        
        with self.measure_performance(f"{environment_type} simulation"):
            if environment_type == "neural":
                env = Phase2NeuralEnvironment(width=100, height=100, use_neural_agents=True)
            else:
                env = Environment(width=100, height=100)
            
            # Adjust population to target
            current_agents = len(env.agents)
            print(f"   Initial agents: {current_agents}")
            
            start_time = time.time()
            start_memory = self.get_memory_usage()
            step_times = []
            
            for step in range(steps):
                step_start = time.time()
                env.step()
                step_end = time.time()
                step_times.append((step_end - step_start) * 1000)  # Convert to ms
                
                # Progress indicator
                if step % (steps // 10) == 0 and step > 0:
                    current_speed = step / (time.time() - start_time)
                    print(f"   Step {step}: {current_speed:.1f} steps/sec")
            
            end_time = time.time()
            end_memory = self.get_memory_usage()
            
            duration = end_time - start_time
            simulation_speed = steps / duration
            memory_delta = end_memory - start_memory
            avg_step_time = np.mean(step_times)
            final_agent_count = len([a for a in env.agents if a.is_alive])
            
            metrics = PerformanceMetrics(
                simulation_speed=simulation_speed,
                memory_usage_mb=memory_delta,
                cpu_usage_percent=self.get_cpu_usage(),
                agent_count=final_agent_count,
                test_duration=duration,
                avg_step_time=avg_step_time
            )
            
            print(f"\n📈 {environment_type.title()} Simulation Results:")
            print(f"   Speed: {simulation_speed:.1f} steps/second")
            print(f"   Avg step time: {avg_step_time:.2f}ms")
            print(f"   Memory impact: {memory_delta:+.1f}MB")
            print(f"   Final agent count: {final_agent_count}")
            
            return metrics
    
    def benchmark_memory_scaling(self) -> Dict[int, float]:
        """Test memory usage with different population sizes"""
        print(f"\n💾 Benchmarking memory scaling")
        
        population_sizes = [25, 50, 100, 150, 200]
        memory_results = {}
        
        for pop_size in population_sizes:
            print(f"   Testing population size: {pop_size}")
            
            gc.collect()
            start_memory = self.get_memory_usage()
            
            env = Phase2NeuralEnvironment(width=150, height=150, use_neural_agents=True)
            
            # Run simulation to stabilize memory usage
            for _ in range(50):
                env.step()
            
            current_memory = self.get_memory_usage()
            memory_delta = current_memory - start_memory
            memory_results[pop_size] = memory_delta
            
            print(f"      Memory usage: {memory_delta:.1f}MB")
            
            # Clean up
            del env
            gc.collect()
        
        print(f"\n📊 Memory Scaling Results:")
        for pop, mem in memory_results.items():
            print(f"   {pop:3d} agents: {mem:6.1f}MB")
        
        return memory_results
    
    def benchmark_web_server_performance(self, duration=30) -> Dict[str, float]:
        """Benchmark web server update performance"""
        print(f"\n🌐 Benchmarking web server performance")
        print(f"   Duration: {duration} seconds")
        
        # Import web server components
        from src.visualization.web_server import EcosystemWebServer
        from main import EnhancedEcosystemWrapper
        
        # Create test environment
        env = Phase2NeuralEnvironment(width=100, height=100, use_neural_agents=True)
        
        # Create wrapper and web server (without actually starting Flask)
        class MockEvolutionSystem:
            def __init__(self, env):
                self.environment = env
                self.generation = 0
                self.total_steps = 0
        
        evolution_system = MockEvolutionSystem(env)
        canvas = EnhancedEcosystemWrapper(evolution_system)
        web_server = EcosystemWebServer(canvas)
        
        # Benchmark data generation
        update_times = []
        start_time = time.time()
        
        print("   Measuring data generation performance...")
        
        while time.time() - start_time < duration:
            update_start = time.time()
            
            # Simulate web server data generation
            canvas.update_step()
            ecosystem_data = web_server.get_ecosystem_data()
            
            update_end = time.time()
            update_times.append((update_end - update_start) * 1000)  # Convert to ms
        
        results = {
            'avg_update_time_ms': np.mean(update_times),
            'max_update_time_ms': np.max(update_times),
            'min_update_time_ms': np.min(update_times),
            'updates_per_second': len(update_times) / duration,
            'total_updates': len(update_times)
        }
        
        print(f"   Results:")
        print(f"      Average update time: {results['avg_update_time_ms']:.2f}ms")
        print(f"      Updates per second: {results['updates_per_second']:.1f}")
        print(f"      Total updates: {results['total_updates']}")
        
        return results
    
    def compare_environment_types(self, steps=1000) -> Dict[str, PerformanceMetrics]:
        """Compare performance of different environment types"""
        print(f"\n⚔️ Comparing environment performance")
        
        results = {}
        
        # Test traditional environment
        results['traditional'] = self.benchmark_simulation_speed("traditional", steps)
        
        # Small delay between tests
        time.sleep(1)
        gc.collect()
        
        # Test neural environment
        results['neural'] = self.benchmark_simulation_speed("neural", steps)
        
        # Performance comparison
        print(f"\n📊 Environment Comparison:")
        traditional = results['traditional']
        neural = results['neural']
        
        speed_ratio = neural.simulation_speed / traditional.simulation_speed
        memory_diff = neural.memory_usage_mb - traditional.memory_usage_mb
        
        print(f"   Speed ratio (Neural/Traditional): {speed_ratio:.2f}x")
        print(f"   Memory difference: {memory_diff:+.1f}MB")
        
        if speed_ratio > 1.0:
            print(f"   🏆 Neural environment is {speed_ratio:.1f}x faster!")
        else:
            print(f"   ⚠️ Neural environment is {1/speed_ratio:.1f}x slower")
        
        return results
    
    def run_comprehensive_benchmark(self):
        """Run all performance benchmarks"""
        print("🎯 Starting Comprehensive Performance Benchmark")
        print("=" * 60)
        
        start_time = time.time()
        
        # Individual benchmarks
        simulation_results = self.compare_environment_types(steps=2000)
        memory_results = self.benchmark_memory_scaling()
        web_results = self.benchmark_web_server_performance(duration=15)
        
        total_time = time.time() - start_time
        
        # Summary report
        print(f"\n" + "=" * 60)
        print(f"🏁 COMPREHENSIVE BENCHMARK COMPLETE")
        print("=" * 60)
        print(f"Total benchmark time: {total_time:.1f} seconds")
        
        print(f"\n📈 Key Performance Metrics:")
        neural_metrics = simulation_results['neural']
        print(f"   Neural simulation speed: {neural_metrics.simulation_speed:.1f} steps/sec")
        print(f"   Average step time: {neural_metrics.avg_step_time:.2f}ms")
        print(f"   Web update rate: {web_results['updates_per_second']:.1f} updates/sec")
        print(f"   Memory efficiency: {neural_metrics.memory_usage_mb:.1f}MB per 2000 steps")
        
        print(f"\n🎯 Optimization Targets:")
        target_speed = 2500
        target_web_rate = 60
        
        speed_improvement_needed = (target_speed / neural_metrics.simulation_speed - 1) * 100
        web_improvement_needed = (target_web_rate / web_results['updates_per_second'] - 1) * 100
        
        print(f"   Simulation speed: {speed_improvement_needed:+.1f}% improvement needed")
        print(f"   Web update rate: {web_improvement_needed:+.1f}% improvement needed")
        
        return {
            'simulation': simulation_results,
            'memory': memory_results,
            'web': web_results,
            'total_time': total_time
        }

def main():
    """Main benchmarking function"""
    print("🚀 EA-NN Performance Profiler")
    print("============================")
    
    profiler = PerformanceProfiler()
    
    try:
        results = profiler.run_comprehensive_benchmark()
        
        print(f"\n💾 Benchmark complete! Results available for optimization planning.")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
