#!/usr/bin/env python3
"""
GPU Acceleration Standalone Demo
==============================

Standalone demonstration of GPU acceleration capabilities.
Shows neural network acceleration, spatial operations, and ecosystem simulation
without module dependencies.
"""

import time
import numpy as np

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    print(f"✅ PyTorch {torch.__version__} loaded")
    print(f"CUDA Available: {'✅ Yes' if CUDA_AVAILABLE else '❌ No (CPU fallback)'}")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    print("❌ PyTorch not available - using NumPy fallback")

class SimpleGPUNeuralNetwork:
    """Simple GPU-accelerated neural network for demonstration"""
    
    def __init__(self, input_size=10, hidden_size=15, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
            self.network = self._create_torch_network()
            print(f"Neural network on: {self.device}")
        else:
            self.device = 'numpy'
            self.network = self._create_numpy_network()
            print("Neural network on: NumPy CPU")
    
    def _create_torch_network(self):
        """Create PyTorch neural network"""
        class Net(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(Net, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, output_size),
                    nn.Tanh()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        net = Net(self.input_size, self.hidden_size, self.output_size)
        net = net.to(self.device)
        net.eval()
        return net
    
    def _create_numpy_network(self):
        """Create NumPy neural network fallback"""
        np.random.seed(42)
        return {
            'W1': np.random.randn(self.input_size, self.hidden_size) * 0.5,
            'b1': np.zeros((1, self.hidden_size)),
            'W2': np.random.randn(self.hidden_size, self.output_size) * 0.5,
            'b2': np.zeros((1, self.output_size))
        }
    
    def process_batch(self, inputs):
        """Process batch of inputs"""
        if TORCH_AVAILABLE:
            input_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.network(input_tensor)
            return outputs.cpu().numpy()
        else:
            # NumPy fallback
            h1 = np.tanh(np.dot(inputs, self.network['W1']) + self.network['b1'])
            outputs = np.tanh(np.dot(h1, self.network['W2']) + self.network['b2'])
            return outputs

class SimpleGPUSpatialOps:
    """Simple GPU spatial operations for demonstration"""
    
    def __init__(self):
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
            print(f"Spatial operations on: {self.device}")
        else:
            self.device = 'numpy'
            print("Spatial operations on: NumPy CPU")
    
    def calculate_distances(self, positions):
        """Calculate pairwise distances"""
        if TORCH_AVAILABLE and len(positions) > 50:
            pos_tensor = torch.tensor(positions, dtype=torch.float32).to(self.device)
            diff = pos_tensor.unsqueeze(1) - pos_tensor.unsqueeze(0)
            distances = torch.sqrt(torch.sum(diff**2, dim=2))
            return distances.cpu().numpy()
        else:
            # NumPy fallback
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            return distances

class SimpleEcosystemDemo:
    """Simple ecosystem demonstration with GPU acceleration"""
    
    def __init__(self, num_agents=200):
        self.num_agents = num_agents
        self.neural_net = SimpleGPUNeuralNetwork()
        self.spatial_ops = SimpleGPUSpatialOps()
        
        # Initialize agents
        np.random.seed(42)
        self.agents = {
            'positions': np.random.uniform(0, 100, (num_agents, 2)),
            'energies': np.random.uniform(0.5, 1.0, num_agents),
            'ages': np.random.randint(0, 100, num_agents),
            'speeds': np.random.uniform(0.5, 2.0, num_agents)
        }
        
        print(f"Ecosystem initialized with {num_agents} agents")
    
    def step(self):
        """Perform one simulation step"""
        step_start = time.time()
        
        # Prepare neural network inputs
        inputs = np.column_stack([
            self.agents['positions'][:, 0] / 100.0,  # Normalized x
            self.agents['positions'][:, 1] / 100.0,  # Normalized y
            self.agents['energies'],
            self.agents['ages'] / 100.0,
            self.agents['speeds'],
            np.random.uniform(0, 1, self.num_agents),  # Random input
            np.random.uniform(0, 1, self.num_agents),  # Random input
            np.random.uniform(0, 1, self.num_agents),  # Random input
            np.random.uniform(0, 1, self.num_agents),  # Random input
            np.ones(self.num_agents)  # Bias
        ])
        
        # Neural network processing
        neural_start = time.time()
        outputs = self.neural_net.process_batch(inputs)
        neural_time = time.time() - neural_start
        
        # Spatial operations
        spatial_start = time.time()
        if self.num_agents <= 300:  # Avoid memory issues for demo
            distances = self.spatial_ops.calculate_distances(self.agents['positions'])
        else:
            distances = None
        spatial_time = time.time() - spatial_start
        
        # Update agents based on neural outputs
        movements = outputs[:, :2] * 2.0  # First 2 outputs as movement
        energy_changes = outputs[:, 2] * 0.1  # Third output as energy change
        
        # Apply updates
        self.agents['positions'] += movements
        self.agents['positions'] = np.clip(self.agents['positions'], 0, 100)
        self.agents['energies'] -= np.abs(energy_changes)
        self.agents['energies'] = np.clip(self.agents['energies'], 0, 1)
        self.agents['ages'] += 1
        
        total_time = time.time() - step_start
        
        return {
            'total_time': total_time,
            'neural_time': neural_time,
            'spatial_time': spatial_time,
            'distance_matrix': distances is not None,
            'agent_count': self.num_agents
        }

def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    
    print("\n🚀 GPU ACCELERATION PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    # Test different population sizes
    population_sizes = [50, 100, 200, 300]
    
    results = {}
    
    for pop_size in population_sizes:
        print(f"\n🧪 Testing population size: {pop_size}")
        print("-" * 50)
        
        ecosystem = SimpleEcosystemDemo(num_agents=pop_size)
        
        # Warm up
        for _ in range(3):
            ecosystem.step()
        
        # Benchmark
        step_times = []
        neural_times = []
        spatial_times = []
        
        benchmark_steps = 20
        start_time = time.time()
        
        for i in range(benchmark_steps):
            result = ecosystem.step()
            step_times.append(result['total_time'])
            neural_times.append(result['neural_time'])
            spatial_times.append(result['spatial_time'])
            
            if i % 5 == 0:
                print(f"  Step {i}: {result['total_time']:.3f}s")
        
        total_benchmark_time = time.time() - start_time
        
        # Calculate statistics
        avg_step_time = np.mean(step_times)
        avg_neural_time = np.mean(neural_times)
        avg_spatial_time = np.mean(spatial_times)
        processing_rate = pop_size / avg_step_time
        
        results[pop_size] = {
            'avg_step_time': avg_step_time,
            'avg_neural_time': avg_neural_time,
            'avg_spatial_time': avg_spatial_time,
            'processing_rate': processing_rate,
            'total_benchmark_time': total_benchmark_time
        }
        
        print(f"  Average step time: {avg_step_time:.3f}s")
        print(f"  Neural processing: {avg_neural_time:.3f}s")
        print(f"  Spatial operations: {avg_spatial_time:.3f}s")
        print(f"  Processing rate: {processing_rate:.1f} agents/sec")
    
    return results

def display_benchmark_results(results):
    """Display comprehensive benchmark results"""
    
    print("\n📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"{'Population':<12} {'Step Time':<12} {'Neural':<10} {'Spatial':<10} {'Rate':<15}")
    print("-" * 70)
    
    for pop_size, data in results.items():
        print(f"{pop_size:<12} {data['avg_step_time']:.3f}s{'':<6} "
              f"{data['avg_neural_time']:.3f}s{'':<4} {data['avg_spatial_time']:.3f}s{'':<4} "
              f"{data['processing_rate']:.1f} agents/sec")
    
    # Calculate scalability
    if len(results) >= 2:
        sizes = list(results.keys())
        rates = [results[size]['processing_rate'] for size in sizes]
        
        print(f"\n📈 SCALABILITY ANALYSIS")
        print("-" * 70)
        
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[0]
            rate_ratio = rates[i] / rates[0]
            efficiency = rate_ratio / size_ratio
            
            print(f"Population {sizes[0]} -> {sizes[i]}: "
                  f"{size_ratio:.1f}x size, {rate_ratio:.2f}x throughput, "
                  f"{efficiency:.2f} efficiency")

def main():
    """Main demonstration function"""
    
    print("🚀 GPU ACCELERATION STANDALONE DEMONSTRATION")
    print("High-Performance Evolutionary Neural Networks")
    print("=" * 80)
    
    # System information
    print(f"🔧 SYSTEM CONFIGURATION")
    print("-" * 40)
    
    if TORCH_AVAILABLE:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {CUDA_AVAILABLE}")
        if CUDA_AVAILABLE:
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
        else:
            print("Running in CPU mode with PyTorch acceleration")
    else:
        print("Running in NumPy CPU fallback mode")
    
    # Run performance benchmark
    results = run_performance_benchmark()
    
    # Display results
    display_benchmark_results(results)
    
    # Final assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT")
    print("=" * 70)
    
    # Get the largest population result
    max_pop = max(results.keys())
    max_rate = results[max_pop]['processing_rate']
    
    if max_rate > 1000:
        print("🚀 EXCELLENT: System delivers high-performance processing")
        print(f"   Peak rate: {max_rate:.1f} agents/sec")
        print("   Ready for large-scale evolutionary simulations")
    elif max_rate > 500:
        print("✅ GOOD: System provides solid performance")
        print(f"   Peak rate: {max_rate:.1f} agents/sec")
        print("   Suitable for medium-scale research")
    else:
        print("💡 BASIC: System provides baseline performance")
        print(f"   Peak rate: {max_rate:.1f} agents/sec")
        print("   Good for initial development and testing")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 70)
    
    if CUDA_AVAILABLE:
        print("🎯 GPU acceleration is active and optimized")
        print("✅ Ready for maximum-performance research")
    elif TORCH_AVAILABLE:
        print("🎯 CPU acceleration is active with PyTorch")
        print("💡 Consider GPU hardware for additional speedup")
    else:
        print("🎯 NumPy CPU processing is active")
        print("💡 Install PyTorch for enhanced performance")
    
    print("\n✅ GPU Acceleration Demonstration Complete!")
    print("🚀 System ready for evolutionary neural network research!")
    
    return results

if __name__ == "__main__":
    results = main()
