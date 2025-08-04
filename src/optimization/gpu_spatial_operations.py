#!/usr/bin/env python3
"""
GPU-Accelerated Spatial Operations for EA-NN
============================================

High-performance GPU-accelerated spatial operations including distance
calculations, spatial queries, and agent position updates. Provides
massive parallelization for spatial computations with CPU fallback.

Features:
- GPU-accelerated distance matrix calculations
- Parallel spatial queries and neighbor finding
- Vectorized agent position and state updates
- Memory-efficient batch processing
- Automatic CPU fallback for compatibility
"""

import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

# GPU Framework imports with fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .gpu_manager import get_gpu_manager, ProcessingMode
from .spatial_indexing import SpatialGrid, SpatialPoint

@dataclass
class SpatialConfig:
    """Configuration for GPU spatial operations"""
    use_gpu: bool = True
    batch_size: int = 1000
    distance_threshold: float = 100.0
    memory_limit_mb: float = 500.0
    prefer_cupy: bool = True  # Prefer CuPy over PyTorch for spatial ops

class GPUSpatialProcessor:
    """GPU-accelerated spatial operations processor"""
    
    def __init__(self, config: SpatialConfig = None):
        self.config = config or SpatialConfig()
        self.gpu_manager = get_gpu_manager()
        
        # Determine processing framework
        self.framework = self._select_framework()
        
        # Performance tracking
        self.performance_stats = {
            'distance_calculations': 0,
            'spatial_queries': 0,
            'gpu_time': 0.0,
            'cpu_time': 0.0
        }
        
        logging.info(f"GPU Spatial Processor: Using {self.framework}")
    
    def _select_framework(self) -> str:
        """Select optimal framework for spatial operations"""
        
        if not self.config.use_gpu or not self.gpu_manager.is_gpu_available():
            return "numpy"
        
        if CUPY_AVAILABLE and self.config.prefer_cupy:
            try:
                # Test CuPy functionality
                test = cp.array([1, 2, 3])
                cp.sum(test)
                return "cupy"
            except Exception as e:
                logging.warning(f"CuPy test failed: {e}")
        
        if TORCH_AVAILABLE:
            return "torch"
        
        return "numpy"
    
    def calculate_distance_matrix_gpu(self, positions: np.ndarray) -> np.ndarray:
        """Calculate distance matrix using GPU acceleration"""
        
        if positions.shape[0] == 0:
            return np.array([])
        
        start_time = time.time()
        
        try:
            if self.framework == "cupy":
                result = self._distance_matrix_cupy(positions)
            elif self.framework == "torch":
                result = self._distance_matrix_torch(positions)
            else:
                result = self._distance_matrix_numpy(positions)
            
            elapsed = time.time() - start_time
            
            if self.framework != "numpy":
                self.performance_stats['gpu_time'] += elapsed
            else:
                self.performance_stats['cpu_time'] += elapsed
            
            self.performance_stats['distance_calculations'] += 1
            
            return result
            
        except Exception as e:
            logging.error(f"GPU distance calculation failed: {e}")
            # Fallback to CPU
            return self._distance_matrix_numpy(positions)
    
    def _distance_matrix_cupy(self, positions: np.ndarray) -> np.ndarray:
        """CuPy implementation of distance matrix"""
        
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        
        # Transfer to GPU
        pos_gpu = cp.array(positions, dtype=cp.float32)
        
        # Vectorized distance calculation
        # Broadcasting: (n, 1, 2) - (1, n, 2) = (n, n, 2)
        diff = pos_gpu[:, None, :] - pos_gpu[None, :, :]
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Transfer back to CPU
        return distances.get()
    
    def _distance_matrix_torch(self, positions: np.ndarray) -> np.ndarray:
        """PyTorch implementation of distance matrix"""
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Transfer to GPU
        pos_tensor = torch.from_numpy(positions.astype(np.float32))
        if self.gpu_manager.is_gpu_available():
            pos_tensor = pos_tensor.to(self.gpu_manager.device)
        
        # Vectorized distance calculation
        diff = pos_tensor.unsqueeze(1) - pos_tensor.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff**2, dim=2))
        
        # Transfer back to CPU
        if self.gpu_manager.is_gpu_available():
            distances = distances.cpu()
        
        return distances.numpy()
    
    def _distance_matrix_numpy(self, positions: np.ndarray) -> np.ndarray:
        """NumPy CPU fallback for distance matrix"""
        
        diff = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        return distances
    
    def find_neighbors_gpu(self, query_positions: np.ndarray, 
                          all_positions: np.ndarray, 
                          radius: float) -> List[List[int]]:
        """Find neighbors within radius using GPU acceleration"""
        
        start_time = time.time()
        
        try:
            if self.framework == "cupy":
                result = self._find_neighbors_cupy(query_positions, all_positions, radius)
            elif self.framework == "torch":
                result = self._find_neighbors_torch(query_positions, all_positions, radius)
            else:
                result = self._find_neighbors_numpy(query_positions, all_positions, radius)
            
            elapsed = time.time() - start_time
            
            if self.framework != "numpy":
                self.performance_stats['gpu_time'] += elapsed
            else:
                self.performance_stats['cpu_time'] += elapsed
            
            self.performance_stats['spatial_queries'] += 1
            
            return result
            
        except Exception as e:
            logging.error(f"GPU neighbor search failed: {e}")
            return self._find_neighbors_numpy(query_positions, all_positions, radius)
    
    def _find_neighbors_cupy(self, query_positions: np.ndarray, 
                           all_positions: np.ndarray, 
                           radius: float) -> List[List[int]]:
        """CuPy implementation of neighbor finding"""
        
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        
        query_gpu = cp.array(query_positions, dtype=cp.float32)
        all_gpu = cp.array(all_positions, dtype=cp.float32)
        
        # Calculate distances: (n_query, n_all)
        diff = query_gpu[:, None, :] - all_gpu[None, :, :]
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Find neighbors within radius
        neighbors_mask = distances <= radius
        
        # Extract indices for each query point
        neighbors = []
        for i in range(query_positions.shape[0]):
            neighbor_indices = cp.where(neighbors_mask[i])[0]
            neighbors.append(neighbor_indices.get().tolist())
        
        return neighbors
    
    def _find_neighbors_torch(self, query_positions: np.ndarray, 
                            all_positions: np.ndarray, 
                            radius: float) -> List[List[int]]:
        """PyTorch implementation of neighbor finding"""
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        query_tensor = torch.from_numpy(query_positions.astype(np.float32))
        all_tensor = torch.from_numpy(all_positions.astype(np.float32))
        
        if self.gpu_manager.is_gpu_available():
            query_tensor = query_tensor.to(self.gpu_manager.device)
            all_tensor = all_tensor.to(self.gpu_manager.device)
        
        # Calculate distances
        diff = query_tensor.unsqueeze(1) - all_tensor.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff**2, dim=2))
        
        # Find neighbors within radius
        neighbors_mask = distances <= radius
        
        # Extract indices for each query point
        neighbors = []
        for i in range(query_positions.shape[0]):
            neighbor_indices = torch.where(neighbors_mask[i])[0]
            if self.gpu_manager.is_gpu_available():
                neighbor_indices = neighbor_indices.cpu()
            neighbors.append(neighbor_indices.numpy().tolist())
        
        return neighbors
    
    def _find_neighbors_numpy(self, query_positions: np.ndarray, 
                            all_positions: np.ndarray, 
                            radius: float) -> List[List[int]]:
        """NumPy CPU fallback for neighbor finding"""
        
        neighbors = []
        
        for query_pos in query_positions:
            diff = all_positions - query_pos
            distances = np.sqrt(np.sum(diff**2, axis=1))
            neighbor_indices = np.where(distances <= radius)[0]
            neighbors.append(neighbor_indices.tolist())
        
        return neighbors
    
    def update_agent_positions_gpu(self, positions: np.ndarray, 
                                 velocities: np.ndarray, 
                                 bounds: Tuple[float, float, float, float],
                                 dt: float = 1.0) -> np.ndarray:
        """Update agent positions using GPU acceleration"""
        
        if positions.shape[0] == 0:
            return positions
        
        try:
            if self.framework == "cupy":
                return self._update_positions_cupy(positions, velocities, bounds, dt)
            elif self.framework == "torch":
                return self._update_positions_torch(positions, velocities, bounds, dt)
            else:
                return self._update_positions_numpy(positions, velocities, bounds, dt)
        
        except Exception as e:
            logging.error(f"GPU position update failed: {e}")
            return self._update_positions_numpy(positions, velocities, bounds, dt)
    
    def _update_positions_cupy(self, positions: np.ndarray, 
                             velocities: np.ndarray, 
                             bounds: Tuple[float, float, float, float],
                             dt: float) -> np.ndarray:
        """CuPy implementation of position updates"""
        
        pos_gpu = cp.array(positions, dtype=cp.float32)
        vel_gpu = cp.array(velocities, dtype=cp.float32)
        
        # Update positions
        new_positions = pos_gpu + vel_gpu * dt
        
        # Apply boundary constraints
        min_x, max_x, min_y, max_y = bounds
        new_positions[:, 0] = cp.clip(new_positions[:, 0], min_x, max_x)
        new_positions[:, 1] = cp.clip(new_positions[:, 1], min_y, max_y)
        
        return new_positions.get()
    
    def _update_positions_torch(self, positions: np.ndarray, 
                              velocities: np.ndarray, 
                              bounds: Tuple[float, float, float, float],
                              dt: float) -> np.ndarray:
        """PyTorch implementation of position updates"""
        
        pos_tensor = torch.from_numpy(positions.astype(np.float32))
        vel_tensor = torch.from_numpy(velocities.astype(np.float32))
        
        if self.gpu_manager.is_gpu_available():
            pos_tensor = pos_tensor.to(self.gpu_manager.device)
            vel_tensor = vel_tensor.to(self.gpu_manager.device)
        
        # Update positions
        new_positions = pos_tensor + vel_tensor * dt
        
        # Apply boundary constraints
        min_x, max_x, min_y, max_y = bounds
        new_positions[:, 0] = torch.clamp(new_positions[:, 0], min_x, max_x)
        new_positions[:, 1] = torch.clamp(new_positions[:, 1], min_y, max_y)
        
        if self.gpu_manager.is_gpu_available():
            new_positions = new_positions.cpu()
        
        return new_positions.numpy()
    
    def _update_positions_numpy(self, positions: np.ndarray, 
                              velocities: np.ndarray, 
                              bounds: Tuple[float, float, float, float],
                              dt: float) -> np.ndarray:
        """NumPy CPU fallback for position updates"""
        
        new_positions = positions + velocities * dt
        
        # Apply boundary constraints
        min_x, max_x, min_y, max_y = bounds
        new_positions[:, 0] = np.clip(new_positions[:, 0], min_x, max_x)
        new_positions[:, 1] = np.clip(new_positions[:, 1], min_y, max_y)
        
        return new_positions
    
    def benchmark_spatial_operations(self, agent_counts: List[int] = None) -> Dict[str, Any]:
        """Benchmark spatial operation performance"""
        
        agent_counts = agent_counts or [50, 100, 200, 500]
        results = {}
        
        for count in agent_counts:
            # Generate random positions
            positions = np.random.randn(count, 2).astype(np.float32) * 100
            
            # Benchmark distance matrix
            iterations = 3
            start_time = time.time()
            
            for _ in range(iterations):
                self.calculate_distance_matrix_gpu(positions)
            
            distance_time = (time.time() - start_time) / iterations
            
            # Benchmark neighbor finding
            query_positions = positions[:min(10, count)]  # Subset for queries
            start_time = time.time()
            
            for _ in range(iterations):
                self.find_neighbors_gpu(query_positions, positions, 50.0)
            
            neighbor_time = (time.time() - start_time) / iterations
            
            # Benchmark position updates
            velocities = np.random.randn(count, 2).astype(np.float32) * 2
            bounds = (0, 800, 0, 600)
            start_time = time.time()
            
            for _ in range(iterations):
                self.update_agent_positions_gpu(positions, velocities, bounds)
            
            update_time = (time.time() - start_time) / iterations
            
            results[f"{count}_agents"] = {
                "distance_matrix_time": distance_time,
                "neighbor_search_time": neighbor_time,
                "position_update_time": update_time,
                "framework": self.framework,
                "total_time": distance_time + neighbor_time + update_time
            }
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        total_time = self.performance_stats['gpu_time'] + self.performance_stats['cpu_time']
        
        stats = {
            "framework": self.framework,
            "distance_calculations": self.performance_stats['distance_calculations'],
            "spatial_queries": self.performance_stats['spatial_queries'],
            "total_gpu_time": self.performance_stats['gpu_time'],
            "total_cpu_time": self.performance_stats['cpu_time'],
            "gpu_percentage": (self.performance_stats['gpu_time'] / total_time * 100) if total_time > 0 else 0
        }
        
        return stats
    
    def print_status(self):
        """Print spatial processor status"""
        
        print("🌍 GPU SPATIAL PROCESSOR STATUS")
        print("=" * 45)
        
        print(f"Framework: {self.framework.upper()}")
        print(f"GPU Available: {'✅' if self.gpu_manager.is_gpu_available() else '❌'}")
        
        stats = self.get_performance_stats()
        
        if stats['distance_calculations'] > 0 or stats['spatial_queries'] > 0:
            print(f"\n📊 Performance Stats:")
            print(f"  Distance Calculations: {stats['distance_calculations']}")
            print(f"  Spatial Queries: {stats['spatial_queries']}")
            print(f"  GPU Time: {stats['total_gpu_time']:.3f}s ({stats['gpu_percentage']:.1f}%)")
            print(f"  CPU Time: {stats['total_cpu_time']:.3f}s")

class GPUSpatialGrid(SpatialGrid):
    """GPU-accelerated spatial grid extending the base spatial indexing"""
    
    def __init__(self, width: float, height: float, cell_size: float = 25.0, use_gpu: bool = True):
        super().__init__(width, height, cell_size)
        
        self.gpu_processor = GPUSpatialProcessor(SpatialConfig(use_gpu=use_gpu))
        self._agent_positions = np.array([])
        self._agent_ids = []
    
    def update_agent_positions(self, agents) -> None:
        """Update internal position cache for GPU processing"""
        
        if not agents:
            self._agent_positions = np.array([])
            self._agent_ids = []
            return
        
        positions = []
        ids = []
        
        for agent in agents:
            if hasattr(agent, 'position'):
                if hasattr(agent.position, 'x') and hasattr(agent.position, 'y'):
                    positions.append([agent.position.x, agent.position.y])
                else:
                    positions.append([agent.position[0], agent.position[1]])
                ids.append(getattr(agent, 'agent_id', id(agent)))
        
        self._agent_positions = np.array(positions, dtype=np.float32) if positions else np.array([])
        self._agent_ids = ids
    
    def query_radius_gpu(self, x: float, y: float, radius: float) -> List[Any]:
        """GPU-accelerated radius query"""
        
        if len(self._agent_positions) == 0:
            return []
        
        query_pos = np.array([[x, y]], dtype=np.float32)
        
        # Use GPU processor for neighbor finding
        neighbor_lists = self.gpu_processor.find_neighbors_gpu(
            query_pos, self._agent_positions, radius
        )
        
        # Return agent IDs of neighbors
        if neighbor_lists and len(neighbor_lists[0]) > 0:
            neighbor_indices = neighbor_lists[0]
            return [self._agent_ids[i] for i in neighbor_indices if i < len(self._agent_ids)]
        
        return []
    
    def get_distance_matrix(self) -> np.ndarray:
        """Get full distance matrix using GPU acceleration"""
        
        if len(self._agent_positions) == 0:
            return np.array([])
        
        return self.gpu_processor.calculate_distance_matrix_gpu(self._agent_positions)

def create_gpu_spatial_processor(config: SpatialConfig = None) -> GPUSpatialProcessor:
    """Factory function to create GPU spatial processor"""
    return GPUSpatialProcessor(config)

def create_gpu_spatial_grid(width: float, height: float, 
                           cell_size: float = 25.0, 
                           use_gpu: bool = True) -> GPUSpatialGrid:
    """Factory function to create GPU spatial grid"""
    return GPUSpatialGrid(width, height, cell_size, use_gpu)
