"""
GPU Accelerated Ecosystem - Practical Implementation
==================================================

Production-ready GPU acceleration that seamlessly falls back to CPU when GPU unavailable.
Integrates with existing EA-NN optimization framework for maximum performance.

Key Features:
- Automatic GPU/CPU detection and fallback
- PyTorch-based neural network acceleration  
- Vectorized spatial operations
- Hybrid processing for optimal resource utilization
- Full compatibility with existing ecosystem

Author: AI Assistant
Date: August 2025
"""

import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Optional GPU acceleration imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GPUPerformanceStats:
    """Performance statistics for GPU acceleration monitoring"""
    
    neural_network_time: float = 0.0
    spatial_operations_time: float = 0.0
    total_gpu_time: float = 0.0
    total_cpu_time: float = 0.0
    agents_processed: int = 0
    gpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    
    def get_total_speedup(self) -> float:
        """Calculate overall speedup achieved"""
        if self.total_cpu_time > 0:
            return self.total_cpu_time / max(self.total_gpu_time, 0.001)
        return 1.0
    
    def get_processing_rate(self) -> float:
        """Get agents processed per second"""
        total_time = self.total_gpu_time + self.total_cpu_time
        if total_time > 0:
            return self.agents_processed / total_time
        return 0.0

class GPUAcceleratedNeuralNetwork:
    """
    GPU-accelerated neural network with automatic CPU fallback
    
    Provides seamless acceleration for agent neural networks using PyTorch.
    Automatically falls back to CPU when GPU unavailable.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 use_gpu: bool = True, device: str = None):
        """
        Initialize GPU-accelerated neural network
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of hidden neurons  
            output_size: Number of output neurons
            use_gpu: Whether to attempt GPU acceleration
            device: Specific device to use (auto-detected if None)
        """
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # GPU availability check
        self.gpu_available = (TORCH_AVAILABLE and torch.cuda.is_available() and use_gpu)
        
        if device is None:
            self.device = torch.device('cuda:0' if self.gpu_available else 'cpu') if TORCH_AVAILABLE else 'cpu'
        else:
            self.device = torch.device(device) if TORCH_AVAILABLE else 'cpu'
        
        # Initialize neural network
        if TORCH_AVAILABLE:
            self.network = self._create_torch_network()
        else:
            self.network = self._create_numpy_network()
        
        logger.info(f"Neural network initialized on {self.device}")
    
    def _create_torch_network(self):
        """Create PyTorch neural network"""
        
        class TorchNeuralNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(TorchNeuralNet, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, output_size),
                    nn.Tanh()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        network = TorchNeuralNet(self.input_size, self.hidden_size, self.output_size)
        network = network.to(self.device)
        network.eval()  # Set to evaluation mode
        
        return network
    
    def _create_numpy_network(self) -> Dict:
        """Create NumPy-based neural network fallback"""
        
        # Initialize random weights
        np.random.seed(42)
        
        network = {
            'W1': np.random.randn(self.input_size, self.hidden_size) * 0.5,
            'b1': np.zeros((1, self.hidden_size)),
            'W2': np.random.randn(self.hidden_size, self.hidden_size) * 0.5,
            'b2': np.zeros((1, self.hidden_size)),
            'W3': np.random.randn(self.hidden_size, self.output_size) * 0.5,
            'b3': np.zeros((1, self.output_size))
        }
        
        return network
    
    def forward_batch(self, inputs: np.ndarray) -> np.ndarray:
        """
        Process batch of inputs through neural network
        
        Args:
            inputs: Input batch (batch_size, input_size)
            
        Returns:
            Output batch (batch_size, output_size)
        """
        
        if TORCH_AVAILABLE and isinstance(self.network, nn.Module):
            return self._forward_torch_batch(inputs)
        else:
            return self._forward_numpy_batch(inputs)
    
    def _forward_torch_batch(self, inputs: np.ndarray) -> np.ndarray:
        """PyTorch batch processing"""
        
        # Convert to tensor and move to device
        input_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output_tensor = self.network(input_tensor)
        
        # Convert back to numpy
        return output_tensor.cpu().numpy()
    
    def _forward_numpy_batch(self, inputs: np.ndarray) -> np.ndarray:
        """NumPy batch processing fallback"""
        
        def tanh_activation(x):
            return np.tanh(x)
        
        # Forward pass through layers
        h1 = tanh_activation(np.dot(inputs, self.network['W1']) + self.network['b1'])
        h2 = tanh_activation(np.dot(h1, self.network['W2']) + self.network['b2'])
        output = tanh_activation(np.dot(h2, self.network['W3']) + self.network['b3'])
        
        return output
    
    def forward_single(self, inputs: np.ndarray) -> np.ndarray:
        """Process single input through neural network"""
        batch_input = inputs.reshape(1, -1)
        batch_output = self.forward_batch(batch_input)
        return batch_output[0]

class GPUAcceleratedSpatialOperations:
    """
    GPU-accelerated spatial operations for agent interactions
    
    Provides high-performance distance calculations, neighbor finding,
    and spatial queries using GPU acceleration where available.
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize GPU spatial operations"""
        
        self.gpu_available = (TORCH_AVAILABLE and torch.cuda.is_available() and use_gpu)
        self.device = torch.device('cuda:0' if self.gpu_available else 'cpu') if TORCH_AVAILABLE else 'cpu'
        
        logger.info(f"Spatial operations initialized on {self.device}")
    
    def calculate_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distance matrix between all positions
        
        Args:
            positions: Agent positions (n_agents, 2)
            
        Returns:
            Distance matrix (n_agents, n_agents)
        """
        
        if TORCH_AVAILABLE and self.gpu_available and len(positions) > 50:
            return self._calculate_distance_matrix_gpu(positions)
        else:
            return self._calculate_distance_matrix_cpu(positions)
    
    def _calculate_distance_matrix_gpu(self, positions: np.ndarray) -> np.ndarray:
        """GPU-accelerated distance matrix calculation"""
        
        # Convert to tensor
        pos_tensor = torch.tensor(positions, dtype=torch.float32).to(self.device)
        
        # Vectorized distance calculation
        diff = pos_tensor.unsqueeze(1) - pos_tensor.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff**2, dim=2))
        
        return distances.cpu().numpy()
    
    def _calculate_distance_matrix_cpu(self, positions: np.ndarray) -> np.ndarray:
        """CPU distance matrix calculation"""
        
        # Vectorized NumPy calculation
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        return distances
    
    def find_neighbors_within_radius(self, positions: np.ndarray, 
                                   query_positions: np.ndarray, 
                                   radius: float) -> List[List[int]]:
        """
        Find neighbors within radius for each query position
        
        Args:
            positions: All agent positions (n_agents, 2)
            query_positions: Query positions (n_queries, 2)  
            radius: Search radius
            
        Returns:
            List of neighbor indices for each query
        """
        
        if TORCH_AVAILABLE and self.gpu_available and len(positions) > 100:
            return self._find_neighbors_gpu(positions, query_positions, radius)
        else:
            return self._find_neighbors_cpu(positions, query_positions, radius)
    
    def _find_neighbors_gpu(self, positions: np.ndarray, 
                           query_positions: np.ndarray, 
                           radius: float) -> List[List[int]]:
        """GPU neighbor finding"""
        
        pos_tensor = torch.tensor(positions, dtype=torch.float32).to(self.device)
        query_tensor = torch.tensor(query_positions, dtype=torch.float32).to(self.device)
        
        # Calculate distances from queries to all positions
        diff = query_tensor.unsqueeze(1) - pos_tensor.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff**2, dim=2))
        
        # Find neighbors within radius
        neighbors = []
        radius_tensor = torch.tensor(radius).to(self.device)
        
        for i in range(len(query_positions)):
            within_radius = torch.where(distances[i] <= radius_tensor)[0]
            neighbors.append(within_radius.cpu().numpy().tolist())
        
        return neighbors
    
    def _find_neighbors_cpu(self, positions: np.ndarray, 
                           query_positions: np.ndarray, 
                           radius: float) -> List[List[int]]:
        """CPU neighbor finding"""
        
        neighbors = []
        
        for query_pos in query_positions:
            diff = positions - query_pos
            distances = np.sqrt(np.sum(diff**2, axis=1))
            within_radius = np.where(distances <= radius)[0]
            neighbors.append(within_radius.tolist())
        
        return neighbors

class GPUAcceleratedEcosystemPractical:
    """
    Production-ready GPU-accelerated ecosystem
    
    Integrates GPU acceleration into existing EA-NN framework with:
    - Automatic hardware detection and fallback
    - Hybrid CPU/GPU processing  
    - Performance monitoring and optimization
    - Full compatibility with existing code
    """
    
    def __init__(self, width: int, height: int, 
                 use_gpu: bool = True, 
                 gpu_threshold: int = 100,
                 neural_config: Dict = None):
        """
        Initialize GPU-accelerated ecosystem
        
        Args:
            width: Ecosystem width
            height: Ecosystem height
            use_gpu: Whether to attempt GPU acceleration
            gpu_threshold: Minimum agents for GPU processing
            neural_config: Neural network configuration
        """
        
        self.width = width
        self.height = height
        self.use_gpu = use_gpu
        self.gpu_threshold = gpu_threshold
        
        # Default neural network configuration
        if neural_config is None:
            neural_config = {
                'input_size': 10,
                'hidden_size': 15,
                'output_size': 4
            }
        
        self.neural_config = neural_config
        
        # Initialize components
        self.gpu_neural_net = GPUAcceleratedNeuralNetwork(**neural_config, use_gpu=use_gpu)
        self.gpu_spatial_ops = GPUAcceleratedSpatialOperations(use_gpu=use_gpu)
        
        # Performance tracking
        self.stats = GPUPerformanceStats()
        self.agents = []
        
        # GPU availability info
        self.gpu_available = (TORCH_AVAILABLE and torch.cuda.is_available() and use_gpu)
        
        logger.info(f"GPU Accelerated Ecosystem initialized")
        logger.info(f"GPU Available: {self.gpu_available}")
        logger.info(f"GPU Threshold: {gpu_threshold} agents")
        
    def add_agent(self, agent: Dict) -> None:
        """Add agent to ecosystem"""
        self.agents.append(agent)
    
    def step_optimized(self) -> Dict[str, Any]:
        """
        Perform optimized ecosystem step with GPU acceleration
        
        Returns:
            Performance statistics and step results
        """
        
        start_time = time.time()
        
        if len(self.agents) == 0:
            return {"step_time": 0.0, "agents": 0, "mode": "idle"}
        
        # Determine processing mode
        processing_mode = self._determine_processing_mode()
        
        # Process agents
        if processing_mode == "gpu":
            results = self._process_agents_gpu()
        else:
            results = self._process_agents_cpu()
        
        # Update performance stats
        step_time = time.time() - start_time
        self.stats.agents_processed = len(self.agents)
        
        if processing_mode == "gpu":
            self.stats.total_gpu_time += step_time
        else:
            self.stats.total_cpu_time += step_time
        
        # Return step results
        return {
            "step_time": step_time,
            "agents": len(self.agents),
            "mode": processing_mode,
            "speedup": self.stats.get_total_speedup(),
            "processing_rate": self.stats.get_processing_rate(),
            **results
        }
    
    def _determine_processing_mode(self) -> str:
        """Determine optimal processing mode (GPU vs CPU)"""
        
        if not self.gpu_available:
            return "cpu"
        
        if len(self.agents) >= self.gpu_threshold:
            return "gpu"
        
        return "cpu"
    
    def _process_agents_gpu(self) -> Dict[str, Any]:
        """Process agents using GPU acceleration"""
        
        neural_start = time.time()
        
        # Batch process neural networks
        agent_inputs = np.array([self._get_agent_inputs(agent) for agent in self.agents])
        agent_outputs = self.gpu_neural_net.forward_batch(agent_inputs)
        
        neural_time = time.time() - neural_start
        
        spatial_start = time.time()
        
        # GPU spatial operations
        agent_positions = np.array([[agent['x'], agent['y']] for agent in self.agents])
        
        # Calculate distances for interactions
        if len(self.agents) > 10:
            distance_matrix = self.gpu_spatial_ops.calculate_distance_matrix(agent_positions)
        else:
            distance_matrix = None
        
        spatial_time = time.time() - spatial_start
        
        # Update agents with GPU results
        for i, agent in enumerate(self.agents):
            self._update_agent_with_output(agent, agent_outputs[i])
        
        # Update performance stats
        self.stats.neural_network_time += neural_time
        self.stats.spatial_operations_time += spatial_time
        
        return {
            "neural_time": neural_time,
            "spatial_time": spatial_time,
            "distance_matrix_computed": distance_matrix is not None
        }
    
    def _process_agents_cpu(self) -> Dict[str, Any]:
        """Process agents using CPU"""
        
        neural_start = time.time()
        
        # Process neural networks individually
        for agent in self.agents:
            inputs = self._get_agent_inputs(agent)
            outputs = self.gpu_neural_net.forward_single(inputs)
            self._update_agent_with_output(agent, outputs)
        
        neural_time = time.time() - neural_start
        
        spatial_start = time.time()
        
        # CPU spatial operations (simplified)
        agent_positions = np.array([[agent['x'], agent['y']] for agent in self.agents])
        
        # Simple neighbor checking
        for agent in self.agents:
            agent['neighbors'] = self._find_nearby_agents_simple(agent, agent_positions)
        
        spatial_time = time.time() - spatial_start
        
        return {
            "neural_time": neural_time,
            "spatial_time": spatial_time,
            "distance_matrix_computed": False
        }
    
    def _get_agent_inputs(self, agent: Dict) -> np.ndarray:
        """Get neural network inputs for agent"""
        
        # Basic sensory inputs
        inputs = np.array([
            agent.get('x', 0.0) / self.width,  # Normalized position
            agent.get('y', 0.0) / self.height,
            agent.get('energy', 1.0),
            agent.get('age', 0.0) / 1000.0,
            agent.get('vision_range', 50.0) / 100.0,
            agent.get('speed', 1.0),
            len(agent.get('neighbors', [])) / 10.0,  # Normalized neighbor count
            agent.get('last_food_time', 0.0) / 100.0,
            agent.get('reproduction_readiness', 0.0),
            1.0  # Bias input
        ])
        
        return inputs
    
    def _update_agent_with_output(self, agent: Dict, outputs: np.ndarray) -> None:
        """Update agent based on neural network output"""
        
        # Interpret neural network outputs
        move_x = outputs[0] * 2.0  # Movement in x direction
        move_y = outputs[1] * 2.0  # Movement in y direction
        energy_consumption = abs(outputs[2])  # Energy usage
        reproduction_desire = outputs[3]  # Reproduction behavior
        
        # Apply agent updates
        agent['x'] = np.clip(agent['x'] + move_x, 0, self.width)
        agent['y'] = np.clip(agent['y'] + move_y, 0, self.height)
        agent['energy'] = max(0, agent['energy'] - energy_consumption * 0.1)
        agent['reproduction_readiness'] = max(0, min(1, reproduction_desire))
        
        # Update age
        agent['age'] = agent.get('age', 0) + 1
    
    def _find_nearby_agents_simple(self, agent: Dict, all_positions: np.ndarray) -> List[int]:
        """Simple neighbor finding for CPU mode"""
        
        agent_pos = np.array([agent['x'], agent['y']])
        distances = np.sqrt(np.sum((all_positions - agent_pos)**2, axis=1))
        
        vision_range = agent.get('vision_range', 50.0)
        nearby = np.where(distances <= vision_range)[0]
        
        return nearby.tolist()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        return {
            "gpu_available": self.gpu_available,
            "total_agents": len(self.agents),
            "processing_mode": self._determine_processing_mode(),
            "total_speedup": self.stats.get_total_speedup(),
            "processing_rate": self.stats.get_processing_rate(),
            "neural_network_time": self.stats.neural_network_time,
            "spatial_operations_time": self.stats.spatial_operations_time,
            "gpu_utilization": self.stats.gpu_utilization,
            "memory_usage_mb": self.stats.memory_usage_mb
        }

def create_demo_ecosystem() -> GPUAcceleratedEcosystemPractical:
    """Create demonstration ecosystem with sample agents"""
    
    ecosystem = GPUAcceleratedEcosystemPractical(
        width=800,
        height=600,
        use_gpu=True,
        gpu_threshold=50
    )
    
    # Add sample agents
    np.random.seed(42)
    
    for i in range(200):
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
    
    return ecosystem

def benchmark_gpu_performance() -> Dict[str, Any]:
    """Benchmark GPU vs CPU performance"""
    
    print("🚀 GPU ACCELERATION PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    ecosystem = create_demo_ecosystem()
    
    # Warm up
    for _ in range(5):
        ecosystem.step_optimized()
    
    # Benchmark multiple steps
    n_steps = 50
    start_time = time.time()
    
    step_results = []
    for i in range(n_steps):
        result = ecosystem.step_optimized()
        step_results.append(result)
        
        if i % 10 == 0:
            print(f"Step {i}: {result['step_time']:.3f}s, Mode: {result['mode']}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    avg_step_time = total_time / n_steps
    total_agents = len(ecosystem.agents)
    processing_rate = total_agents / avg_step_time
    
    # Performance summary
    summary = ecosystem.get_performance_summary()
    
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Steps: {n_steps}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Step Time: {avg_step_time:.3f}s")
    print(f"Total Agents: {total_agents}")
    print(f"Processing Rate: {processing_rate:.1f} agents/sec")
    print(f"GPU Available: {summary['gpu_available']}")
    print(f"Processing Mode: {summary['processing_mode']}")
    print(f"Overall Speedup: {summary['total_speedup']:.2f}x")
    
    return {
        "total_time": total_time,
        "avg_step_time": avg_step_time,
        "processing_rate": processing_rate,
        "step_results": step_results,
        "performance_summary": summary
    }

if __name__ == "__main__":
    # Run performance benchmark
    benchmark_results = benchmark_gpu_performance()
    
    print("\n✅ GPU Acceleration System Ready!")
    print("🚀 High-performance evolutionary neural networks enabled!")
