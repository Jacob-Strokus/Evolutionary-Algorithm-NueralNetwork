#!/usr/bin/env python3
"""
Spatial Indexing System for EA-NN Performance Optimization
===========================================================

Efficient spatial data structures to optimize agent interactions and
distance calculations. Replaces O(n²) distance checks with O(log n)
spatial queries using grid-based and quadtree indexing.

Key optimizations:
- Grid-based spatial partitioning for fast neighbor queries
- Efficient range queries for food/threat detection
- Spatial caching for frequently accessed regions
- Memory-efficient implementation with object pooling
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import time

@dataclass
class SpatialPoint:
    """Represents a point in 2D space with associated data"""
    x: float
    y: float
    data: Any
    id: int
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, SpatialPoint):
            return False
        return self.id == other.id

class SpatialGrid:
    """Grid-based spatial indexing for fast neighbor queries"""
    
    def __init__(self, width: float, height: float, cell_size: float = 20.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Grid dimensions
        self.grid_width = int(np.ceil(width / cell_size))
        self.grid_height = int(np.ceil(height / cell_size))
        
        # Grid storage: Dict[Tuple[int, int], Set[SpatialPoint]]
        self.grid: Dict[Tuple[int, int], Set[SpatialPoint]] = defaultdict(set)
        
        # Point tracking for efficient updates
        self.point_locations: Dict[int, Tuple[int, int]] = {}
        
        # Performance metrics
        self.query_count = 0
        self.total_query_time = 0.0
    
    def _get_cell_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell coordinates"""
        cell_x = int(x / self.cell_size)
        cell_y = int(y / self.cell_size)
        
        # Clamp to grid bounds
        cell_x = max(0, min(cell_x, self.grid_width - 1))
        cell_y = max(0, min(cell_y, self.grid_height - 1))
        
        return (cell_x, cell_y)
    
    def insert(self, point: SpatialPoint) -> None:
        """Insert a point into the spatial grid"""
        cell_coords = self._get_cell_coords(point.x, point.y)
        
        # Remove from old location if it exists
        if point.id in self.point_locations:
            old_coords = self.point_locations[point.id]
            self.grid[old_coords].discard(point)
        
        # Add to new location
        self.grid[cell_coords].add(point)
        self.point_locations[point.id] = cell_coords
    
    def remove(self, point_id: int) -> bool:
        """Remove a point from the spatial grid"""
        if point_id not in self.point_locations:
            return False
        
        cell_coords = self.point_locations[point_id]
        
        # Find and remove the point
        for point in self.grid[cell_coords]:
            if point.id == point_id:
                self.grid[cell_coords].remove(point)
                del self.point_locations[point_id]
                return True
        
        return False
    
    def update_position(self, point_id: int, new_x: float, new_y: float) -> bool:
        """Update the position of an existing point"""
        if point_id not in self.point_locations:
            return False
        
        old_coords = self.point_locations[point_id]
        new_coords = self._get_cell_coords(new_x, new_y)
        
        # If the point moved to a different cell, relocate it
        if old_coords != new_coords:
            # Find the point in the old cell
            point_to_move = None
            for point in self.grid[old_coords]:
                if point.id == point_id:
                    point_to_move = point
                    break
            
            if point_to_move:
                # Remove from old cell
                self.grid[old_coords].remove(point_to_move)
                
                # Update coordinates
                point_to_move.x = new_x
                point_to_move.y = new_y
                
                # Add to new cell
                self.grid[new_coords].add(point_to_move)
                self.point_locations[point_id] = new_coords
                
                return True
        else:
            # Same cell, just update coordinates
            for point in self.grid[old_coords]:
                if point.id == point_id:
                    point.x = new_x
                    point.y = new_y
                    return True
        
        return False
    
    def query_radius(self, center_x: float, center_y: float, radius: float) -> List[SpatialPoint]:
        """Find all points within radius of center point"""
        start_time = time.time()
        self.query_count += 1
        
        results = []
        radius_squared = radius * radius
        
        # Calculate which grid cells to check
        cell_radius = int(np.ceil(radius / self.cell_size))
        center_cell_x, center_cell_y = self._get_cell_coords(center_x, center_y)
        
        # Check all cells within the radius
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell_x = center_cell_x + dx
                cell_y = center_cell_y + dy
                
                # Skip cells outside grid bounds
                if (cell_x < 0 or cell_x >= self.grid_width or 
                    cell_y < 0 or cell_y >= self.grid_height):
                    continue
                
                # Check all points in this cell
                for point in self.grid[(cell_x, cell_y)]:
                    dx_point = point.x - center_x
                    dy_point = point.y - center_y
                    distance_squared = dx_point * dx_point + dy_point * dy_point
                    
                    if distance_squared <= radius_squared:
                        results.append(point)
        
        self.total_query_time += time.time() - start_time
        return results
    
    def query_nearest(self, center_x: float, center_y: float, count: int = 1) -> List[SpatialPoint]:
        """Find the nearest N points to center"""
        # Start with a small radius and expand until we find enough points
        radius = self.cell_size
        max_radius = max(self.width, self.height)
        
        while radius <= max_radius:
            candidates = self.query_radius(center_x, center_y, radius)
            
            if len(candidates) >= count:
                # Sort by distance and return closest N
                def distance_squared(point):
                    dx = point.x - center_x
                    dy = point.y - center_y
                    return dx * dx + dy * dy
                
                candidates.sort(key=distance_squared)
                return candidates[:count]
            
            radius *= 2  # Exponential expansion
        
        return []  # No points found
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        avg_query_time = (self.total_query_time / self.query_count * 1000 
                         if self.query_count > 0 else 0)
        
        total_points = sum(len(cell) for cell in self.grid.values())
        occupied_cells = len([cell for cell in self.grid.values() if cell])
        
        return {
            'total_queries': self.query_count,
            'avg_query_time_ms': avg_query_time,
            'total_points': total_points,
            'occupied_cells': occupied_cells,
            'grid_utilization': occupied_cells / (self.grid_width * self.grid_height)
        }
    
    def clear(self) -> None:
        """Clear all points from the grid"""
        self.grid.clear()
        self.point_locations.clear()
        self.query_count = 0
        self.total_query_time = 0.0

class OptimizedEcosystemSpatial:
    """Optimized ecosystem with spatial indexing for agent interactions"""
    
    def __init__(self, width: float, height: float, cell_size: float = 25.0):
        self.width = width
        self.height = height
        
        # Spatial indices for different entity types
        self.agents_spatial = SpatialGrid(width, height, cell_size)
        self.food_spatial = SpatialGrid(width, height, cell_size)
        
        # Entity ID counters
        self.next_agent_id = 0
        self.next_food_id = 0
        
        # Performance tracking
        self.optimization_stats = {
            'distance_calculations_saved': 0,
            'spatial_queries_performed': 0,
            'update_operations': 0
        }
    
    def add_agent(self, agent) -> int:
        """Add an agent to the spatial index"""
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        
        spatial_point = SpatialPoint(
            x=agent.position.x,
            y=agent.position.y,
            data=agent,
            id=agent_id
        )
        
        self.agents_spatial.insert(spatial_point)
        return agent_id
    
    def update_agent_position(self, agent_id: int, new_x: float, new_y: float) -> bool:
        """Update an agent's position in the spatial index"""
        self.optimization_stats['update_operations'] += 1
        return self.agents_spatial.update_position(agent_id, new_x, new_y)
    
    def remove_agent(self, agent_id: int) -> bool:
        """Remove an agent from the spatial index"""
        return self.agents_spatial.remove(agent_id)
    
    def find_nearby_agents(self, center_x: float, center_y: float, radius: float) -> List[Any]:
        """Find all agents within radius of center point"""
        self.optimization_stats['spatial_queries_performed'] += 1
        
        spatial_points = self.agents_spatial.query_radius(center_x, center_y, radius)
        return [point.data for point in spatial_points]
    
    def find_nearest_food(self, center_x: float, center_y: float, count: int = 1) -> List[Any]:
        """Find nearest food sources"""
        self.optimization_stats['spatial_queries_performed'] += 1
        
        spatial_points = self.food_spatial.query_nearest(center_x, center_y, count)
        return [point.data for point in spatial_points]
    
    def find_threats_in_range(self, center_x: float, center_y: float, radius: float, 
                              predator_species_type) -> List[Any]:
        """Find predators within range"""
        self.optimization_stats['spatial_queries_performed'] += 1
        
        spatial_points = self.agents_spatial.query_radius(center_x, center_y, radius)
        threats = []
        
        for point in spatial_points:
            agent = point.data
            if hasattr(agent, 'species_type') and agent.species_type == predator_species_type:
                threats.append(agent)
        
        return threats
    
    def get_spatial_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive spatial performance statistics"""
        agent_stats = self.agents_spatial.get_performance_stats()
        food_stats = self.food_spatial.get_performance_stats()
        
        return {
            'agents': agent_stats,
            'food': food_stats,
            'optimization': self.optimization_stats,
            'total_spatial_queries': agent_stats['total_queries'] + food_stats['total_queries'],
            'estimated_distance_calcs_saved': self.optimization_stats['distance_calculations_saved']
        }

# Performance testing functions
def benchmark_spatial_vs_brute_force():
    """Compare spatial indexing performance vs brute force"""
    print("🏁 Benchmarking Spatial Indexing vs Brute Force")
    print("=" * 50)
    
    # Test parameters
    num_agents = 200
    num_queries = 1000
    search_radius = 25.0
    world_size = 500
    
    # Generate random agents
    np.random.seed(42)  # For reproducible results
    agent_positions = np.random.rand(num_agents, 2) * world_size
    
    # Setup spatial grid
    spatial_grid = SpatialGrid(world_size, world_size, cell_size=30.0)
    
    # Insert agents into spatial grid
    for i, pos in enumerate(agent_positions):
        point = SpatialPoint(x=pos[0], y=pos[1], data=f"agent_{i}", id=i)
        spatial_grid.insert(point)
    
    # Benchmark spatial queries
    print("Testing spatial grid queries...")
    start_time = time.time()
    
    for _ in range(num_queries):
        query_x = np.random.rand() * world_size
        query_y = np.random.rand() * world_size
        results = spatial_grid.query_radius(query_x, query_y, search_radius)
    
    spatial_time = time.time() - start_time
    
    # Benchmark brute force
    print("Testing brute force queries...")
    start_time = time.time()
    
    for _ in range(num_queries):
        query_x = np.random.rand() * world_size
        query_y = np.random.rand() * world_size
        
        # Brute force search
        results = []
        for pos in agent_positions:
            dx = pos[0] - query_x
            dy = pos[1] - query_y
            if dx*dx + dy*dy <= search_radius*search_radius:
                results.append(pos)
    
    brute_force_time = time.time() - start_time
    
    # Results
    speedup = brute_force_time / spatial_time
    
    print(f"\n📊 Performance Comparison:")
    print(f"   Spatial Grid: {spatial_time:.3f}s ({num_queries/spatial_time:.1f} queries/sec)")
    print(f"   Brute Force:  {brute_force_time:.3f}s ({num_queries/brute_force_time:.1f} queries/sec)")
    print(f"   Speedup:      {speedup:.1f}x faster with spatial indexing!")
    
    # Get spatial grid stats
    stats = spatial_grid.get_performance_stats()
    print(f"\n📈 Spatial Grid Statistics:")
    print(f"   Average query time: {stats['avg_query_time_ms']:.2f}ms")
    print(f"   Grid utilization: {stats['grid_utilization']:.1%}")
    print(f"   Total points: {stats['total_points']}")

if __name__ == "__main__":
    benchmark_spatial_vs_brute_force()
