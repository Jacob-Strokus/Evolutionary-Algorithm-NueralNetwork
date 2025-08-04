#!/usr/bin/env python3
"""
GPU-Accelerated Neural Networks for EA-NN
=========================================

High-performance GPU-accelerated neural network implementations for
evolutionary agents. Provides batch processing, memory optimization,
and seamless CPU fallback for maximum performance and compatibility.

Features:
- GPU-accelerated neural network forward passes
- Batch processing for multiple agents simultaneously
- Mixed precision training for memory efficiency
- Dynamic memory management and optimization
- Automatic CPU fallback when GPU unavailable
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# GPU Framework imports with fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .gpu_manager import get_gpu_manager, ProcessingMode

@dataclass
class NeuralNetworkConfig:
    """Configuration for GPU neural networks"""
    input_size: int = 10
    hidden_sizes: List[int] = None
    output_size: int = 4
    activation: str = 'tanh'
    use_batch_norm: bool = False
    dropout_rate: float = 0.0
    mixed_precision: bool = True
    batch_size_threshold: int = 32

class GPUNeuralNetworkBase(ABC):
    """Abstract base class for GPU neural networks"""
    
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        pass
    
    @abstractmethod
    def batch_forward(self, batch_inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Batch forward pass for multiple inputs"""
        pass

class TorchGPUNeuralNetwork(GPUNeuralNetworkBase):
    """PyTorch-based GPU neural network implementation"""
    
    def __init__(self, config: NeuralNetworkConfig):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for GPU neural networks")
        
        self.config = config
        self.gpu_manager = get_gpu_manager()
        
        # Set default hidden sizes if not provided
        if config.hidden_sizes is None:
            config.hidden_sizes = [20, 15]
        
        # Build network architecture
        self.network = self._build_network()
        
        # Move to GPU if available
        if self.gpu_manager.is_gpu_available():
            self.network = self.network.to(self.gpu_manager.device)
            if config.mixed_precision and self.gpu_manager.capabilities.supports_mixed_precision:
                # Enable automatic mixed precision
                self.use_amp = True
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.use_amp = False
        else:
            self.use_amp = False
        
        # Set to evaluation mode (no training)
        self.network.eval()
        
    def _build_network(self) -> nn.Module:
        """Build neural network architecture"""
        
        layers = []
        sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]
        
        for i in range(len(sizes) - 1):
            # Linear layer
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            
            # Batch normalization (if enabled and not last layer)
            if self.config.use_batch_norm and i < len(sizes) - 2:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            
            # Activation function (not on last layer)
            if i < len(sizes) - 2:
                if self.config.activation == 'tanh':
                    layers.append(nn.Tanh())
                elif self.config.activation == 'relu':
                    layers.append(nn.ReLU())
                elif self.config.activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                else:
                    layers.append(nn.Tanh())  # Default
                
                # Dropout (if enabled)
                if self.config.dropout_rate > 0:
                    layers.append(nn.Dropout(self.config.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Single forward pass"""
        
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs, dtype=np.float32)
        
        # Convert to tensor
        x = torch.from_numpy(inputs.astype(np.float32))
        
        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Move to GPU if available
        if self.gpu_manager.is_gpu_available():
            x = x.to(self.gpu_manager.device)
        
        # Forward pass
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.network(x)
            else:
                output = self.network(x)
        
        # Convert back to numpy
        if self.gpu_manager.is_gpu_available():
            output = output.cpu()
        
        result = output.numpy()
        
        # Remove batch dimension if added
        if result.shape[0] == 1:
            result = result.squeeze(0)
        
        return result
    
    def batch_forward(self, batch_inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Batch forward pass for multiple inputs"""
        
        if len(batch_inputs) == 0:
            return []
        
        # Convert to tensors and stack
        tensor_inputs = []
        for inputs in batch_inputs:
            if not isinstance(inputs, np.ndarray):
                inputs = np.array(inputs, dtype=np.float32)
            tensor_inputs.append(torch.from_numpy(inputs.astype(np.float32)))
        
        # Stack into batch
        batch_tensor = torch.stack(tensor_inputs)
        
        # Move to GPU if available
        if self.gpu_manager.is_gpu_available():
            batch_tensor = batch_tensor.to(self.gpu_manager.device)
        
        # Batch forward pass
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    batch_output = self.network(batch_tensor)
            else:
                batch_output = self.network(batch_tensor)
        
        # Convert back to numpy and split
        if self.gpu_manager.is_gpu_available():
            batch_output = batch_output.cpu()
        
        batch_numpy = batch_output.numpy()
        
        # Split back into individual outputs
        return [batch_numpy[i] for i in range(len(batch_inputs))]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage for this network"""
        
        if not self.gpu_manager.is_gpu_available():
            return {"gpu_memory_mb": 0}
        
        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in self.network.parameters())
        
        return {
            "gpu_memory_mb": param_memory / (1024 * 1024),
            "parameter_count": sum(p.numel() for p in self.network.parameters())
        }

class CPUNeuralNetworkFallback(GPUNeuralNetworkBase):
    """CPU fallback implementation for neural networks"""
    
    def __init__(self, config: NeuralNetworkConfig):
        self.config = config
        
        # Set default hidden sizes
        if config.hidden_sizes is None:
            config.hidden_sizes = [20, 15]
        
        # Initialize weights
        self.weights = []
        self.biases = []
        
        sizes = [config.input_size] + config.hidden_sizes + [config.output_size]
        
        for i in range(len(sizes) - 1):
            # Xavier/Glorot initialization
            limit = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            w = np.random.uniform(-limit, limit, (sizes[i], sizes[i + 1])).astype(np.float32)
            b = np.zeros(sizes[i + 1], dtype=np.float32)
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.config.activation == 'tanh':
            return np.tanh(x)
        elif self.config.activation == 'relu':
            return np.maximum(0, x)
        elif self.config.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        else:
            return np.tanh(x)  # Default
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through CPU network"""
        
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs, dtype=np.float32)
        
        x = inputs.copy()
        
        # Forward through layers
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.dot(x, w) + b
            
            # Apply activation (except on last layer)
            if i < len(self.weights) - 1:
                x = self._activation(x)
        
        return x
    
    def batch_forward(self, batch_inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Batch forward pass (sequential on CPU for simplicity)"""
        return [self.forward(inputs) for inputs in batch_inputs]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage for CPU network"""
        total_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        return {
            "cpu_memory_mb": memory_mb,
            "parameter_count": total_params
        }

class GPUNeuralProcessor:
    """High-level GPU neural network processor"""
    
    def __init__(self, config: NeuralNetworkConfig = None):
        self.config = config or NeuralNetworkConfig()
        self.gpu_manager = get_gpu_manager()
        
        # Create appropriate network based on GPU availability
        if self.gpu_manager.is_gpu_available() and TORCH_AVAILABLE:
            try:
                self.network = TorchGPUNeuralNetwork(self.config)
                self.processor_type = "GPU"
                logging.info("GPU Neural Processor: Using PyTorch GPU acceleration")
            except Exception as e:
                logging.warning(f"GPU neural network creation failed: {e}")
                self.network = CPUNeuralNetworkFallback(self.config)
                self.processor_type = "CPU_FALLBACK"
        else:
            self.network = CPUNeuralNetworkFallback(self.config)
            self.processor_type = "CPU"
            logging.info("GPU Neural Processor: Using CPU fallback")
    
    def process_agent(self, agent_state: np.ndarray) -> np.ndarray:
        """Process single agent neural network"""
        return self.network.forward(agent_state)
    
    def process_agent_batch(self, agent_states: List[np.ndarray]) -> List[np.ndarray]:
        """Process batch of agents (GPU accelerated if available)"""
        
        if len(agent_states) == 0:
            return []
        
        # Use batch processing for GPU, or if batch is large enough
        if (self.processor_type == "GPU" or 
            len(agent_states) >= self.config.batch_size_threshold):
            return self.network.batch_forward(agent_states)
        else:
            # Process individually for small batches on CPU
            return [self.network.forward(state) for state in agent_states]
    
    def get_optimal_batch_size(self, agent_count: int) -> int:
        """Get optimal batch size based on GPU memory and agent count"""
        
        if not self.gpu_manager.is_gpu_available():
            return min(agent_count, 16)  # Small batches for CPU
        
        # Estimate memory usage per agent
        memory_per_agent = self.config.input_size * 4  # 4 bytes per float32
        
        # Available GPU memory (keep 20% buffer)
        available_memory = (self.gpu_manager.capabilities.total_memory_gb * 
                          1024 * 1024 * 1024 * 0.8)
        
        # Calculate maximum batch size
        max_batch = int(available_memory / (memory_per_agent * 10))  # Conservative estimate
        
        # Clamp to reasonable range
        optimal_batch = min(max_batch, agent_count, 256)
        optimal_batch = max(optimal_batch, 1)
        
        return optimal_batch
    
    def benchmark_performance(self, agent_counts: List[int] = None) -> Dict[str, Any]:
        """Benchmark neural processing performance"""
        
        agent_counts = agent_counts or [10, 50, 100, 200]
        results = {}
        
        for count in agent_counts:
            # Generate random agent states
            agent_states = [
                np.random.randn(self.config.input_size).astype(np.float32) 
                for _ in range(count)
            ]
            
            # Benchmark batch processing
            import time
            
            iterations = 5
            start_time = time.time()
            
            for _ in range(iterations):
                self.process_agent_batch(agent_states)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / iterations
            agents_per_sec = count / avg_time
            
            results[f"{count}_agents"] = {
                "time_per_batch": avg_time,
                "agents_per_second": agents_per_sec,
                "processor_type": self.processor_type
            }
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        
        info = {
            "processor_type": self.processor_type,
            "network_config": {
                "input_size": self.config.input_size,
                "hidden_sizes": self.config.hidden_sizes,
                "output_size": self.config.output_size,
                "activation": self.config.activation
            },
            "memory_usage": self.network.get_memory_usage()
        }
        
        info.update(self.gpu_manager.get_system_info())
        
        return info
    
    def print_status(self):
        """Print neural processor status"""
        
        print("🧠 GPU NEURAL PROCESSOR STATUS")
        print("=" * 45)
        
        info = self.get_system_info()
        
        print(f"Processor Type: {info['processor_type']}")
        print(f"Network Architecture: {info['network_config']['input_size']} → {info['network_config']['hidden_sizes']} → {info['network_config']['output_size']}")
        print(f"Activation: {info['network_config']['activation']}")
        
        if 'parameter_count' in info['memory_usage']:
            print(f"Parameters: {info['memory_usage']['parameter_count']:,}")
        
        if self.processor_type == "GPU":
            print(f"GPU Memory: {info['memory_usage'].get('gpu_memory_mb', 0):.1f} MB")
        else:
            print(f"CPU Memory: {info['memory_usage'].get('cpu_memory_mb', 0):.1f} MB")

def create_gpu_neural_processor(config: NeuralNetworkConfig = None) -> GPUNeuralProcessor:
    """Factory function to create GPU neural processor"""
    return GPUNeuralProcessor(config)
