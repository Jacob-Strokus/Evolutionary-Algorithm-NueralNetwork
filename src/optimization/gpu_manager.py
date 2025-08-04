#!/usr/bin/env python3
"""
GPU Hardware Detection and Management System
===========================================

Comprehensive GPU hardware detection, device management, and fallback
system for EA-NN GPU acceleration. Handles CUDA availability, memory
management, and graceful degradation to CPU when GPU is unavailable.

Features:
- Automatic GPU detection and capability assessment
- Memory management and optimization
- Dynamic CPU/GPU switching based on workload
- Performance monitoring and thermal management
- Robust fallback to CPU for maximum compatibility
"""

import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

# GPU Framework imports with fallback handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - GPU acceleration disabled")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.info("CuPy not available - using PyTorch for GPU operations")

import numpy as np

class ProcessingMode(Enum):
    """Available processing modes"""
    CPU = "cpu"
    GPU = "gpu"
    HYBRID = "hybrid"
    AUTO = "auto"

@dataclass
class GPUCapabilities:
    """GPU hardware capabilities"""
    device_count: int
    total_memory_gb: float
    compute_capability: Tuple[int, int]
    device_name: str
    cuda_version: str
    supports_mixed_precision: bool
    memory_bandwidth_gbps: float

@dataclass
class GPUConfig:
    """GPU acceleration configuration"""
    mode: ProcessingMode = ProcessingMode.AUTO
    device_id: int = 0
    memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    batch_threshold: int = 50  # Minimum agents for GPU processing
    fallback_enabled: bool = True
    performance_monitoring: bool = True

class GPUManager:
    """Central GPU management and hardware abstraction"""
    
    def __init__(self, config: GPUConfig = None):
        self.config = config or GPUConfig()
        self.capabilities: Optional[GPUCapabilities] = None
        self.device = None
        self.current_mode = ProcessingMode.CPU
        
        # Performance tracking
        self.performance_history = {
            'gpu_times': [],
            'cpu_times': [],
            'memory_usage': []
        }
        
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU hardware and detect capabilities"""
        
        if not TORCH_AVAILABLE:
            logging.info("GPU Manager: PyTorch not available, using CPU mode")
            self.current_mode = ProcessingMode.CPU
            return
        
        if not torch.cuda.is_available():
            logging.info("GPU Manager: CUDA not available, using CPU mode")
            self.current_mode = ProcessingMode.CPU
            return
        
        try:
            # Detect GPU capabilities
            device_count = torch.cuda.device_count()
            if device_count == 0:
                logging.info("GPU Manager: No CUDA devices found")
                self.current_mode = ProcessingMode.CPU
                return
            
            # Use specified device or default to 0
            device_id = min(self.config.device_id, device_count - 1)
            self.device = torch.device(f'cuda:{device_id}')
            
            # Get device properties
            props = torch.cuda.get_device_properties(device_id)
            
            self.capabilities = GPUCapabilities(
                device_count=device_count,
                total_memory_gb=props.total_memory / (1024**3),
                compute_capability=(props.major, props.minor),
                device_name=props.name,
                cuda_version=torch.version.cuda or "Unknown",
                supports_mixed_precision=props.major >= 7,  # Tensor cores available
                memory_bandwidth_gbps=self._estimate_memory_bandwidth(props)
            )
            
            # Set memory management
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(self.config.memory_fraction, device_id)
            
            self.current_mode = ProcessingMode.GPU
            
            logging.info(f"GPU Manager: Initialized GPU {device_id} - {props.name}")
            logging.info(f"GPU Memory: {self.capabilities.total_memory_gb:.1f} GB")
            logging.info(f"Compute Capability: {self.capabilities.compute_capability}")
            
        except Exception as e:
            logging.error(f"GPU initialization failed: {e}")
            self.current_mode = ProcessingMode.CPU
    
    def _estimate_memory_bandwidth(self, props) -> float:
        """Estimate memory bandwidth based on GPU architecture"""
        # Simplified bandwidth estimation
        memory_gb = props.total_memory / (1024**3)
        
        if "RTX" in props.name or "GTX 16" in props.name:
            return memory_gb * 50  # Rough estimate for gaming GPUs
        elif "Tesla" in props.name or "Quadro" in props.name:
            return memory_gb * 80  # Professional GPUs typically have higher bandwidth
        else:
            return memory_gb * 30  # Conservative estimate
    
    def get_optimal_processing_mode(self, agent_count: int, complexity_factor: float = 1.0) -> ProcessingMode:
        """Determine optimal processing mode based on workload"""
        
        if self.current_mode == ProcessingMode.CPU:
            return ProcessingMode.CPU
        
        if self.config.mode != ProcessingMode.AUTO:
            return self.config.mode
        
        # Dynamic decision based on workload
        adjusted_threshold = self.config.batch_threshold * complexity_factor
        
        if agent_count < adjusted_threshold:
            return ProcessingMode.CPU  # GPU overhead not worth it
        elif agent_count < adjusted_threshold * 3:
            return ProcessingMode.HYBRID  # Mix of GPU and CPU
        else:
            return ProcessingMode.GPU  # Full GPU acceleration
    
    def allocate_gpu_memory(self, size_mb: float) -> bool:
        """Allocate GPU memory and check availability"""
        
        if not self.is_gpu_available():
            return False
        
        try:
            # Check available memory
            memory_free = torch.cuda.get_device_properties(self.device).total_memory
            memory_allocated = torch.cuda.memory_allocated(self.device)
            memory_available = (memory_free - memory_allocated) / (1024**2)  # MB
            
            if size_mb > memory_available * 0.9:  # Keep 10% buffer
                logging.warning(f"Insufficient GPU memory: need {size_mb}MB, have {memory_available}MB")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"GPU memory allocation check failed: {e}")
            return False
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if self.is_gpu_available():
            torch.cuda.empty_cache()
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for processing"""
        return (self.current_mode in [ProcessingMode.GPU, ProcessingMode.HYBRID] and 
                self.device is not None and 
                TORCH_AVAILABLE and 
                torch.cuda.is_available())
    
    def benchmark_performance(self, operations: List[str] = None) -> Dict[str, float]:
        """Benchmark GPU vs CPU performance for specific operations"""
        
        operations = operations or ['matrix_multiply', 'distance_calculation', 'neural_forward']
        results = {}
        
        for operation in operations:
            cpu_time = self._benchmark_cpu_operation(operation)
            gpu_time = self._benchmark_gpu_operation(operation) if self.is_gpu_available() else float('inf')
            
            results[operation] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': cpu_time / gpu_time if gpu_time > 0 else 0
            }
        
        return results
    
    def _benchmark_cpu_operation(self, operation: str) -> float:
        """Benchmark CPU operation performance"""
        
        size = 1000
        iterations = 10
        
        if operation == 'matrix_multiply':
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            start = time.time()
            for _ in range(iterations):
                np.dot(a, b)
            return (time.time() - start) / iterations
            
        elif operation == 'distance_calculation':
            positions = np.random.randn(size, 2).astype(np.float32)
            
            start = time.time()
            for _ in range(iterations):
                # Pairwise distance calculation
                diff = positions[:, None, :] - positions[None, :, :]
                np.sqrt(np.sum(diff**2, axis=2))
            return (time.time() - start) / iterations
            
        elif operation == 'neural_forward':
            x = np.random.randn(size, 10).astype(np.float32)
            w1 = np.random.randn(10, 20).astype(np.float32)
            w2 = np.random.randn(20, 5).astype(np.float32)
            
            start = time.time()
            for _ in range(iterations):
                h = np.tanh(np.dot(x, w1))
                np.dot(h, w2)
            return (time.time() - start) / iterations
        
        return 1.0  # Default
    
    def _benchmark_gpu_operation(self, operation: str) -> float:
        """Benchmark GPU operation performance"""
        
        if not self.is_gpu_available():
            return float('inf')
        
        size = 1000
        iterations = 10
        
        try:
            if operation == 'matrix_multiply':
                a = torch.randn(size, size, device=self.device, dtype=torch.float32)
                b = torch.randn(size, size, device=self.device, dtype=torch.float32)
                
                torch.cuda.synchronize()  # Ensure GPU is ready
                start = time.time()
                for _ in range(iterations):
                    torch.mm(a, b)
                torch.cuda.synchronize()  # Wait for completion
                return (time.time() - start) / iterations
                
            elif operation == 'distance_calculation':
                positions = torch.randn(size, 2, device=self.device, dtype=torch.float32)
                
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(iterations):
                    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
                    torch.sqrt(torch.sum(diff**2, dim=2))
                torch.cuda.synchronize()
                return (time.time() - start) / iterations
                
            elif operation == 'neural_forward':
                x = torch.randn(size, 10, device=self.device, dtype=torch.float32)
                w1 = torch.randn(10, 20, device=self.device, dtype=torch.float32)
                w2 = torch.randn(20, 5, device=self.device, dtype=torch.float32)
                
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(iterations):
                    h = torch.tanh(torch.mm(x, w1))
                    torch.mm(h, w2)
                torch.cuda.synchronize()
                return (time.time() - start) / iterations
                
        except Exception as e:
            logging.error(f"GPU benchmark failed for {operation}: {e}")
            return float('inf')
        
        return 1.0  # Default
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        
        info = {
            'gpu_available': self.is_gpu_available(),
            'current_mode': self.current_mode.value,
            'torch_available': TORCH_AVAILABLE,
            'cupy_available': CUPY_AVAILABLE
        }
        
        if self.capabilities:
            info.update({
                'gpu_name': self.capabilities.device_name,
                'gpu_memory_gb': self.capabilities.total_memory_gb,
                'compute_capability': self.capabilities.compute_capability,
                'cuda_version': self.capabilities.cuda_version,
                'mixed_precision_support': self.capabilities.supports_mixed_precision
            })
        
        if self.is_gpu_available():
            info.update({
                'memory_allocated_mb': torch.cuda.memory_allocated(self.device) / (1024**2),
                'memory_cached_mb': torch.cuda.memory_reserved(self.device) / (1024**2)
            })
        
        return info
    
    def print_status(self):
        """Print comprehensive GPU status"""
        
        print("🚀 GPU ACCELERATION STATUS")
        print("=" * 50)
        
        info = self.get_system_info()
        
        print(f"GPU Available: {'✅ Yes' if info['gpu_available'] else '❌ No'}")
        print(f"Current Mode: {info['current_mode'].upper()}")
        print(f"PyTorch: {'✅' if info['torch_available'] else '❌'}")
        print(f"CuPy: {'✅' if info['cupy_available'] else '❌'}")
        
        if self.capabilities:
            print(f"\n🎯 GPU Hardware:")
            print(f"  Device: {info['gpu_name']}")
            print(f"  Memory: {info['gpu_memory_gb']:.1f} GB")
            print(f"  Compute: {info['compute_capability'][0]}.{info['compute_capability'][1]}")
            print(f"  CUDA: {info['cuda_version']}")
            print(f"  Mixed Precision: {'✅' if info['mixed_precision_support'] else '❌'}")
        
        if self.is_gpu_available():
            print(f"\n📊 Memory Usage:")
            print(f"  Allocated: {info['memory_allocated_mb']:.1f} MB")
            print(f"  Cached: {info['memory_cached_mb']:.1f} MB")

# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None

def get_gpu_manager(config: GPUConfig = None) -> GPUManager:
    """Get or create global GPU manager instance"""
    global _gpu_manager
    
    if _gpu_manager is None:
        _gpu_manager = GPUManager(config)
    
    return _gpu_manager

def initialize_gpu_acceleration(config: GPUConfig = None) -> GPUManager:
    """Initialize GPU acceleration system"""
    
    manager = get_gpu_manager(config)
    manager.print_status()
    
    # Run initial benchmarks
    if manager.is_gpu_available():
        print(f"\n🏃 Performance Benchmarks:")
        print("-" * 30)
        
        benchmarks = manager.benchmark_performance(['matrix_multiply', 'distance_calculation'])
        
        for operation, results in benchmarks.items():
            speedup = results['speedup']
            if speedup > 1:
                print(f"  {operation}: {speedup:.1f}x faster on GPU")
            else:
                print(f"  {operation}: CPU faster for this workload")
    
    return manager
