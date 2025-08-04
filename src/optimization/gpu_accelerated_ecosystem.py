#!/usr/bin/env python3
"""
GPU-Accelerated Ecosystem for EA-NN
===================================

Complete GPU-accelerated ecosystem implementation that integrates
neural network acceleration, spatial operations, and hybrid processing
for maximum performance in evolutionary neural network simulations.

Features:
- Hybrid CPU/GPU processing pipeline
- Automatic workload balancing and optimization
- Memory-efficient batch processing
- Performance monitoring and adaptive scaling
- Seamless fallback to CPU-only operation
"""

import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .gpu_manager import get_gpu_manager, ProcessingMode, GPUConfig
from .gpu_neural_networks import GPUNeuralProcessor, NeuralNetworkConfig
from .gpu_spatial_operations import GPUSpatialProcessor, SpatialConfig, GPUSpatialGrid
from .high_performance_ecosystem import HighPerformanceEnvironment, OptimizationConfig
from ..core.ecosystem import Environment, Agent, SpeciesType, Position

@dataclass
class GPUEcosystemConfig:
    """Configuration for GPU-accelerated ecosystem"""
    # GPU settings
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = True
    
    # Processing thresholds
    neural_gpu_threshold: int = 50  # Minimum agents for GPU neural processing
    spatial_gpu_threshold: int = 100  # Minimum agents for GPU spatial processing
    batch_size_neural: int = 64
    batch_size_spatial: int = 256
    
    # Optimization settings
    adaptive_processing: bool = True
    performance_monitoring: bool = True
    memory_optimization: bool = True
    
    # Fallback settings
    cpu_fallback_enabled: bool = True
    fallback_on_memory_error: bool = True

class ProcessingStats:
    """Performance statistics tracking"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.neural_gpu_time = 0.0
        self.neural_cpu_time = 0.0
        self.spatial_gpu_time = 0.0
        self.spatial_cpu_time = 0.0
        self.total_steps = 0
        self.gpu_memory_peak = 0.0
        self.processing_mode_history = []
    
    def add_step(self, neural_time: float, spatial_time: float, 
                 neural_gpu: bool, spatial_gpu: bool, memory_usage: float):
        """Add performance data for a simulation step"""
        
        if neural_gpu:
            self.neural_gpu_time += neural_time
        else:
            self.neural_cpu_time += neural_time
        
        if spatial_gpu:
            self.spatial_gpu_time += spatial_time
        else:
            self.spatial_cpu_time += spatial_time
        
        self.total_steps += 1
        self.gpu_memory_peak = max(self.gpu_memory_peak, memory_usage)
        
        mode = "gpu" if neural_gpu and spatial_gpu else "hybrid" if neural_gpu or spatial_gpu else "cpu"
        self.processing_mode_history.append(mode)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        if self.total_steps == 0:
            return {"total_steps": 0}
        
        total_neural_time = self.neural_gpu_time + self.neural_cpu_time
        total_spatial_time = self.spatial_gpu_time + self.spatial_cpu_time
        total_time = total_neural_time + total_spatial_time
        
        # Calculate mode percentages
        mode_counts = {}
        for mode in self.processing_mode_history:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        mode_percentages = {
            mode: (count / self.total_steps * 100) 
            for mode, count in mode_counts.items()
        }
        
        return {
            "total_steps": self.total_steps,
            "avg_step_time": total_time / self.total_steps if self.total_steps > 0 else 0,
            "neural_gpu_percentage": (self.neural_gpu_time / total_neural_time * 100) if total_neural_time > 0 else 0,
            "spatial_gpu_percentage": (self.spatial_gpu_time / total_spatial_time * 100) if total_spatial_time > 0 else 0,
            "gpu_memory_peak_mb": self.gpu_memory_peak,
            "processing_modes": mode_percentages,
            "performance_breakdown": {
                "neural_gpu_time": self.neural_gpu_time,
                "neural_cpu_time": self.neural_cpu_time,
                "spatial_gpu_time": self.spatial_gpu_time,
                "spatial_cpu_time": self.spatial_cpu_time
            }
        }

class GPUAcceleratedEcosystem(HighPerformanceEnvironment):
    """GPU-accelerated ecosystem with hybrid processing"""
    
    def __init__(self, width: int, height: int, config: GPUEcosystemConfig = None):
        # Initialize base high-performance ecosystem
        base_config = OptimizationConfig(
            enable_spatial_indexing=True,
            enable_object_pooling=True,
            enable_vectorization=True,
            enable_caching=True
        )
        super().__init__(width, height, base_config)
        
        self.gpu_config = config or GPUEcosystemConfig()
        
        # Initialize GPU components
        self._initialize_gpu_components()
        
        # Performance tracking
        self.stats = ProcessingStats()
        self.step_count = 0
        
        # Adaptive processing parameters
        self.recent_performance = []
        self.last_mode_switch = 0
        
        logging.info("GPU Accelerated Ecosystem initialized")
    
    def _initialize_gpu_components(self):
        """Initialize GPU processing components"""
        
        # GPU Manager
        gpu_config = GPUConfig(
            memory_fraction=self.gpu_config.gpu_memory_fraction,
            enable_mixed_precision=self.gpu_config.mixed_precision,
            fallback_enabled=self.gpu_config.cpu_fallback_enabled
        )
        self.gpu_manager = get_gpu_manager(gpu_config)
        
        # Neural Processor
        neural_config = NeuralNetworkConfig(
            input_size=10,  # Standard agent state size
            hidden_sizes=[20, 15],
            output_size=4,  # Movement + action decisions
            mixed_precision=self.gpu_config.mixed_precision,
            batch_size_threshold=self.gpu_config.batch_size_neural
        )
        self.neural_processor = GPUNeuralProcessor(neural_config)
        
        # Spatial Processor
        spatial_config = SpatialConfig(
            use_gpu=self.gpu_config.enable_gpu,
            batch_size=self.gpu_config.batch_size_spatial,
            memory_limit_mb=self.gpu_config.gpu_memory_fraction * 
                           (self.gpu_manager.capabilities.total_memory_gb * 1024 if self.gpu_manager.capabilities else 1024)
        )
        self.spatial_processor = GPUSpatialProcessor(spatial_config)
        
        # GPU Spatial Grid (replaces regular spatial grid)
        if hasattr(self, 'spatial_grid'):
            self.gpu_spatial_grid = GPUSpatialGrid(
                self.width, self.height, 
                cell_size=25.0, 
                use_gpu=self.gpu_config.enable_gpu
            )
    
    def step(self):
        """Enhanced step with GPU acceleration and performance monitoring"""
        
        step_start_time = time.time()
        
        # Determine processing modes for this step
        agent_count = len(self.agents)
        neural_mode, spatial_mode = self._determine_processing_modes(agent_count)
        
        # Update spatial grid with current agent positions
        if hasattr(self, 'gpu_spatial_grid'):
            self.gpu_spatial_grid.update_agent_positions(self.agents)
        
        # Neural processing phase
        neural_start = time.time()
        if agent_count > 0:
            self._process_agents_neural(neural_mode)
        neural_time = time.time() - neural_start
        
        # Spatial operations phase
        spatial_start = time.time()
        if agent_count > 0:
            self._process_spatial_operations(spatial_mode)
        spatial_time = time.time() - spatial_start
        
        # Regular ecosystem update (food, reproduction, etc.)
        super().step()
        
        # Performance tracking
        memory_usage = self._get_gpu_memory_usage()
        self.stats.add_step(
            neural_time, spatial_time,
            neural_mode == "gpu", spatial_mode == "gpu",
            memory_usage
        )
        
        self.step_count += 1
        
        # Adaptive processing adjustment
        if self.gpu_config.adaptive_processing:
            self._update_adaptive_processing(time.time() - step_start_time)
    
    def _determine_processing_modes(self, agent_count: int) -> Tuple[str, str]:
        """Determine optimal processing modes for current step"""
        
        if not self.gpu_config.enable_gpu or not self.gpu_manager.is_gpu_available():
            return "cpu", "cpu"
        
        # Neural processing mode
        if agent_count >= self.gpu_config.neural_gpu_threshold:
            neural_mode = "gpu"
        else:
            neural_mode = "cpu"
        
        # Spatial processing mode
        if agent_count >= self.gpu_config.spatial_gpu_threshold:
            spatial_mode = "gpu"
        else:
            spatial_mode = "cpu"
        
        # Adaptive adjustments based on recent performance
        if self.gpu_config.adaptive_processing and len(self.recent_performance) > 10:
            neural_mode, spatial_mode = self._adaptive_mode_adjustment(
                neural_mode, spatial_mode, agent_count
            )
        
        return neural_mode, spatial_mode
    
    def _process_agents_neural(self, mode: str):
        """Process agent neural networks using specified mode"""
        
        if not self.agents:
            return
        
        try:
            if mode == "gpu":
                self._process_agents_neural_gpu()
            else:
                self._process_agents_neural_cpu()
        
        except Exception as e:
            if self.gpu_config.fallback_on_memory_error:
                logging.warning(f"Neural GPU processing failed, falling back to CPU: {e}")
                self._process_agents_neural_cpu()
            else:
                raise
    
    def _process_agents_neural_gpu(self):
        """GPU-accelerated neural processing"""
        
        # Extract agent states
        agent_states = []
        for agent in self.agents:
            state = self._get_agent_state(agent)
            agent_states.append(state)
        
        # Batch process neural networks
        neural_outputs = self.neural_processor.process_agent_batch(agent_states)
        
        # Apply neural decisions to agents
        for agent, output in zip(self.agents, neural_outputs):
            self._apply_neural_output(agent, output)
    
    def _process_agents_neural_cpu(self):
        """CPU fallback neural processing"""
        
        for agent in self.agents:
            state = self._get_agent_state(agent)
            output = self.neural_processor.process_agent(state)
            self._apply_neural_output(agent, output)
    
    def _process_spatial_operations(self, mode: str):
        """Process spatial operations using specified mode"""
        
        if not self.agents:
            return
        
        try:
            if mode == "gpu":
                self._process_spatial_gpu()
            else:
                self._process_spatial_cpu()
        
        except Exception as e:
            if self.gpu_config.fallback_on_memory_error:
                logging.warning(f"Spatial GPU processing failed, falling back to CPU: {e}")
                self._process_spatial_cpu()
            else:
                raise
    
    def _process_spatial_gpu(self):
        """GPU-accelerated spatial operations"""
        
        # Extract positions and velocities
        positions = []
        velocities = []
        
        for agent in self.agents:
            if hasattr(agent.position, 'x'):
                positions.append([agent.position.x, agent.position.y])
            else:
                positions.append([agent.position[0], agent.position[1]])
            
            # Simple velocity calculation (could be enhanced)
            velocities.append([np.random.randn() * 2, np.random.randn() * 2])
        
        if positions:
            positions = np.array(positions, dtype=np.float32)
            velocities = np.array(velocities, dtype=np.float32)
            
            # GPU-accelerated position updates
            bounds = (0, self.width, 0, self.height)
            new_positions = self.spatial_processor.update_agent_positions_gpu(
                positions, velocities, bounds, dt=0.1
            )
            
            # Update agent positions
            for agent, new_pos in zip(self.agents, new_positions):
                agent.position = Position(new_pos[0], new_pos[1])
    
    def _process_spatial_cpu(self):
        """CPU fallback spatial operations"""
        
        # Use existing spatial indexing system
        if hasattr(self, 'spatial_grid'):
            # Regular spatial grid processing
            pass  # Use existing implementation
    
    def _get_agent_state(self, agent) -> np.ndarray:
        """Extract neural network input state from agent"""
        
        # Basic state representation (can be enhanced)
        state = np.zeros(10, dtype=np.float32)
        
        # Position
        if hasattr(agent.position, 'x'):
            state[0] = agent.position.x / self.width
            state[1] = agent.position.y / self.height
        else:
            state[0] = agent.position[0] / self.width
            state[1] = agent.position[1] / self.height
        
        # Energy
        state[2] = getattr(agent, 'energy', 50.0) / 100.0
        
        # Species
        if hasattr(agent, 'species_type'):
            state[3] = 1.0 if agent.species_type == SpeciesType.HERBIVORE else -1.0
        
        # Random features (placeholder for sensors, etc.)
        state[4:] = np.random.randn(6) * 0.1
        
        return state
    
    def _apply_neural_output(self, agent, output: np.ndarray):
        """Apply neural network output to agent"""
        
        # Simple movement based on neural output
        if len(output) >= 2:
            dx = output[0] * 5.0  # Scale movement
            dy = output[1] * 5.0
            
            if hasattr(agent.position, 'x'):
                new_x = np.clip(agent.position.x + dx, 0, self.width)
                new_y = np.clip(agent.position.y + dy, 0, self.height)
                agent.position = Position(new_x, new_y)
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        
        if not self.gpu_manager.is_gpu_available():
            return 0.0
        
        try:
            import torch
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except:
            return 0.0
    
    def _update_adaptive_processing(self, step_time: float):
        """Update adaptive processing parameters"""
        
        self.recent_performance.append(step_time)
        
        # Keep only recent history
        if len(self.recent_performance) > 20:
            self.recent_performance.pop(0)
        
        # Adjust thresholds based on performance trends
        if len(self.recent_performance) >= 10:
            recent_avg = np.mean(self.recent_performance[-10:])
            older_avg = np.mean(self.recent_performance[:10])
            
            # If performance is degrading, increase thresholds
            if recent_avg > older_avg * 1.2:
                self.gpu_config.neural_gpu_threshold = min(
                    self.gpu_config.neural_gpu_threshold + 10, 200
                )
                self.gpu_config.spatial_gpu_threshold = min(
                    self.gpu_config.spatial_gpu_threshold + 20, 300
                )
    
    def _adaptive_mode_adjustment(self, neural_mode: str, spatial_mode: str, 
                                agent_count: int) -> Tuple[str, str]:
        """Adjust processing modes based on performance history"""
        
        # Simple heuristic: if recent performance is poor, prefer CPU for smaller workloads
        recent_avg = np.mean(self.recent_performance[-5:])
        
        if recent_avg > 0.1:  # If steps are taking > 100ms
            # Increase thresholds temporarily
            neural_threshold = self.gpu_config.neural_gpu_threshold * 1.5
            spatial_threshold = self.gpu_config.spatial_gpu_threshold * 1.5
            
            if agent_count < neural_threshold:
                neural_mode = "cpu"
            if agent_count < spatial_threshold:
                spatial_mode = "cpu"
        
        return neural_mode, spatial_mode
    
    def benchmark_performance(self, steps: int = 100) -> Dict[str, Any]:
        """Benchmark ecosystem performance with different configurations"""
        
        original_config = self.gpu_config
        results = {}
        
        # Test different configurations
        configs = [
            ("CPU_Only", GPUEcosystemConfig(enable_gpu=False)),
            ("GPU_Conservative", GPUEcosystemConfig(
                neural_gpu_threshold=100, spatial_gpu_threshold=200
            )),
            ("GPU_Aggressive", GPUEcosystemConfig(
                neural_gpu_threshold=20, spatial_gpu_threshold=50
            ))
        ]
        
        for config_name, config in configs:
            self.gpu_config = config
            self.stats.reset()
            
            start_time = time.time()
            
            for _ in range(steps):
                self.step()
            
            total_time = time.time() - start_time
            
            results[config_name] = {
                "total_time": total_time,
                "steps_per_second": steps / total_time,
                "avg_step_time": total_time / steps,
                "performance_stats": self.stats.get_summary()
            }
        
        # Restore original configuration
        self.gpu_config = original_config
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            "gpu_ecosystem_config": {
                "enable_gpu": self.gpu_config.enable_gpu,
                "neural_gpu_threshold": self.gpu_config.neural_gpu_threshold,
                "spatial_gpu_threshold": self.gpu_config.spatial_gpu_threshold,
                "adaptive_processing": self.gpu_config.adaptive_processing
            },
            "current_state": {
                "agent_count": len(self.agents),
                "step_count": self.step_count,
                "gpu_memory_usage_mb": self._get_gpu_memory_usage()
            },
            "gpu_manager": self.gpu_manager.get_system_info(),
            "performance_stats": self.stats.get_summary()
        }
        
        return status
    
    def print_status(self):
        """Print comprehensive ecosystem status"""
        
        print("⚡ GPU ACCELERATED ECOSYSTEM STATUS")
        print("=" * 50)
        
        status = self.get_system_status()
        
        print(f"Agents: {status['current_state']['agent_count']}")
        print(f"Steps: {status['current_state']['step_count']}")
        print(f"GPU Enabled: {'✅' if self.gpu_config.enable_gpu else '❌'}")
        
        if status['gpu_manager']['gpu_available']:
            print(f"GPU: {status['gpu_manager']['gpu_name']}")
            print(f"GPU Memory: {status['current_state']['gpu_memory_usage_mb']:.1f} MB")
        
        stats = status['performance_stats']
        if stats['total_steps'] > 0:
            print(f"\n📊 Performance Summary:")
            print(f"  Average Step Time: {stats['avg_step_time']:.3f}s")
            print(f"  Neural GPU Usage: {stats['neural_gpu_percentage']:.1f}%")
            print(f"  Spatial GPU Usage: {stats['spatial_gpu_percentage']:.1f}%")
            
            modes = stats['processing_modes']
            for mode, percentage in modes.items():
                print(f"  {mode.upper()} Mode: {percentage:.1f}%")

def create_gpu_accelerated_ecosystem(width: int, height: int, 
                                   config: GPUEcosystemConfig = None) -> GPUAcceleratedEcosystem:
    """Factory function to create GPU-accelerated ecosystem"""
    return GPUAcceleratedEcosystem(width, height, config)
