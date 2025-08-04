#!/usr/bin/env python3
"""
High-Performance Ecosystem Implementation
==========================================

Optimized ecosystem implementation using spatial indexing, vectorized operations,
and memory-efficient data structures for maximum simulation performance.

Key optimizations:
- Spatial indexing for O(log n) agent interactions
- Vectorized NumPy operations for bulk calculations
- Object pooling to reduce memory allocations
- Efficient data structures and caching
- Configurable performance vs. accuracy trade-offs
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.ecosystem import Environment, Agent, Food, Position, SpeciesType
from src.optimization.spatial_indexing import SpatialGrid, SpatialPoint, OptimizedEcosystemSpatial

@dataclass
class PerformanceConfig:
    """Configuration for performance vs. accuracy trade-offs"""
    use_spatial_indexing: bool = True
    spatial_cell_size: float = 25.0
    enable_vectorized_updates: bool = True
    enable_object_pooling: bool = True
    max_neighbor_distance: float = 50.0
    update_frequency_food: int = 1  # Update food every N steps
    update_frequency_agents: int = 1  # Update agents every N steps
    enable_performance_profiling: bool = False

class AgentPool:
    """Object pool for efficient agent memory management"""
    
    def __init__(self, initial_size: int = 50, species_type: SpeciesType = SpeciesType.HERBIVORE):
        self.species_type = species_type
        self.available: List[Agent] = []
        self.in_use: Set[Agent] = set()
        
        # Pre-allocate agents
        for _ in range(initial_size):
            agent = self._create_fresh_agent()
            self.available.append(agent)
    
    def _create_fresh_agent(self) -> Agent:
        """Create a new agent instance"""
        # Create at random position (will be overridden when used)
        position = Position(50, 50)
        agent_id = int(time.time() * 1000000) % 1000000  # Simple ID generation
        return Agent(self.species_type, position, agent_id)
    
    def get_agent(self) -> Agent:
        """Get an agent from the pool"""
        if self.available:
            agent = self.available.pop()
        else:
            agent = self._create_fresh_agent()
        
        self.in_use.add(agent)
        # Reset agent state manually
        agent.energy = 100.0
        agent.age = 0
        agent.is_alive = True
        agent.reproduction_cooldown = 0
        return agent
    
    def return_agent(self, agent: Agent) -> None:
        """Return an agent to the pool"""
        if agent in self.in_use:
            self.in_use.remove(agent)
            self.available.append(agent)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        return {
            'available': len(self.available),
            'in_use': len(self.in_use),
            'total': len(self.available) + len(self.in_use)
        }

class HighPerformanceEcosystem(Environment):
    """Optimized ecosystem implementation with spatial indexing and vectorization"""
    
    def __init__(self, width: int = 100, height: int = 100, config: Optional[PerformanceConfig] = None):
        # Initialize base environment
        super().__init__(width, height)
        
        # Performance configuration
        self.config = config or PerformanceConfig()
        
        # Spatial indexing system
        if self.config.use_spatial_indexing:
            self.spatial_system = OptimizedEcosystemSpatial(
                width, height, self.config.spatial_cell_size
            )
            self._setup_spatial_indices()
        
        # Object pools for memory efficiency
        if self.config.enable_object_pooling:
            self.herbivore_pool = AgentPool(initial_size=30, species_type=SpeciesType.HERBIVORE)
            self.carnivore_pool = AgentPool(initial_size=15, species_type=SpeciesType.CARNIVORE)
        
        # Vectorized data arrays for batch operations
        if self.config.enable_vectorized_updates:
            self._setup_vectorized_arrays()
        
        # Performance tracking
        self.performance_stats = {
            'total_steps': 0,
            'total_step_time': 0.0,
            'spatial_queries': 0,
            'vectorized_operations': 0,
            'memory_pool_hits': 0
        }
        
        # Step counters for update frequency control
        self.step_counter = 0
    
    def _setup_spatial_indices(self):
        """Initialize spatial indices with current agents and food"""
        if not hasattr(self, 'spatial_system'):
            return
        
        # Add existing agents to spatial index
        for agent in self.agents:
            agent.spatial_id = self.spatial_system.add_agent(agent)
        
        # Add food sources to spatial index (if implemented)
        # This would require extending the spatial system for food
    
    def _setup_vectorized_arrays(self):
        """Setup NumPy arrays for vectorized operations"""
        max_agents = 200  # Maximum expected agent count
        
        # Pre-allocate arrays for agent data
        self.agent_positions = np.zeros((max_agents, 2), dtype=np.float32)
        self.agent_energies = np.zeros(max_agents, dtype=np.float32)
        self.agent_ages = np.zeros(max_agents, dtype=np.int32)
        self.agent_active = np.zeros(max_agents, dtype=bool)
        
        # Mapping from agent to array index
        self.agent_to_index = {}
        self.next_array_index = 0
    
    def _add_agent_to_arrays(self, agent: Agent) -> int:
        """Add agent to vectorized arrays"""
        if not self.config.enable_vectorized_updates:
            return -1
        
        if self.next_array_index >= len(self.agent_positions):
            # Need to expand arrays
            new_size = len(self.agent_positions) * 2
            self.agent_positions = np.resize(self.agent_positions, (new_size, 2))
            self.agent_energies = np.resize(self.agent_energies, new_size)
            self.agent_ages = np.resize(self.agent_ages, new_size)
            self.agent_active = np.resize(self.agent_active, new_size)
        
        index = self.next_array_index
        self.next_array_index += 1
        
        # Initialize array data
        self.agent_positions[index] = [agent.position.x, agent.position.y]
        self.agent_energies[index] = agent.energy
        self.agent_ages[index] = agent.age
        self.agent_active[index] = True
        
        self.agent_to_index[agent] = index
        return index
    
    def _remove_agent_from_arrays(self, agent: Agent):
        """Remove agent from vectorized arrays"""
        if agent not in self.agent_to_index:
            return
        
        index = self.agent_to_index[agent]
        self.agent_active[index] = False
        del self.agent_to_index[agent]
    
    def _vectorized_update_positions(self):
        """Update all agent positions using vectorized operations"""
        if not self.config.enable_vectorized_updates:
            return
        
        # Get active agents mask - use only the indices that are actually used
        max_used_index = min(self.next_array_index, len(self.agent_active))
        if max_used_index <= 0:
            return
            
        active_mask = self.agent_active[:max_used_index]
        
        if not np.any(active_mask):
            return
        
        # Vectorized energy decay - only update active agents
        energy_decay = 0.5
        self.agent_energies[:max_used_index][active_mask] = np.maximum(
            0, self.agent_energies[:max_used_index][active_mask] - energy_decay
        )
        
        # Vectorized age increment
        self.agent_ages[:max_used_index][active_mask] += 1
        
        # Update agent objects from arrays
        for agent, index in self.agent_to_index.items():
            if index < max_used_index and self.agent_active[index]:
                agent.position.x = self.agent_positions[index, 0]
                agent.position.y = self.agent_positions[index, 1]
                agent.energy = self.agent_energies[index]
                agent.age = self.agent_ages[index]
        
        self.performance_stats['vectorized_operations'] += 1
    
    def find_nearest_food_optimized(self, agent: Agent) -> Optional[Food]:
        """Optimized food finding using spatial indexing"""
        if self.config.use_spatial_indexing and hasattr(self, 'spatial_system'):
            # Use spatial indexing for fast food lookup
            nearby_food = self.spatial_system.find_nearest_food(
                agent.position.x, agent.position.y, count=1
            )
            self.performance_stats['spatial_queries'] += 1
            return nearby_food[0] if nearby_food else None
        else:
            # Fallback to traditional method
            return super().find_nearest_food(agent)
    
    def find_nearest_threat_optimized(self, agent: Agent) -> Optional[Agent]:
        """Optimized threat detection using spatial indexing"""
        if self.config.use_spatial_indexing and hasattr(self, 'spatial_system'):
            # Find nearby carnivores for herbivores
            if agent.species_type == SpeciesType.HERBIVORE:
                threats = self.spatial_system.find_threats_in_range(
                    agent.position.x, agent.position.y,
                    self.config.max_neighbor_distance,
                    SpeciesType.CARNIVORE
                )
                self.performance_stats['spatial_queries'] += 1
                return threats[0] if threats else None
            
            return None
        else:
            # Fallback to traditional method
            return super().find_nearest_threat(agent)
    
    def find_nearby_agents_optimized(self, agent: Agent, radius: float) -> List[Agent]:
        """Optimized nearby agent finding"""
        if self.config.use_spatial_indexing and hasattr(self, 'spatial_system'):
            nearby = self.spatial_system.find_nearby_agents(
                agent.position.x, agent.position.y, radius
            )
            self.performance_stats['spatial_queries'] += 1
            # Filter out self
            return [a for a in nearby if a != agent]
        else:
            # Fallback: traditional distance calculation
            nearby = []
            for other in self.agents:
                if other != agent and agent.position.distance_to(other.position) <= radius:
                    nearby.append(other)
            return nearby
    
    def step_optimized(self):
        """Optimized simulation step with performance enhancements"""
        step_start_time = time.time()
        self.step_counter += 1
        
        # Update time step
        self.time_step += 1
        
        # Vectorized updates (if enabled)
        if (self.config.enable_vectorized_updates and 
            self.step_counter % self.config.update_frequency_agents == 0):
            self._vectorized_update_positions()
        
        # Agent behavior updates
        new_agents = []
        agents_to_remove = []
        
        for agent in self.agents:
            if not agent.is_alive:
                agents_to_remove.append(agent)
                continue
            
            # Update spatial index if agent moved
            if (self.config.use_spatial_indexing and hasattr(self, 'spatial_system') and
                hasattr(agent, 'spatial_id')):
                self.spatial_system.update_agent_position(
                    agent.spatial_id, agent.position.x, agent.position.y
                )
            
            # Agent AI behavior (using optimized methods)
            if agent.species_type == SpeciesType.HERBIVORE:
                # Optimized herbivore behavior
                threat = self.find_nearest_threat_optimized(agent)
                if threat and agent.position.distance_to(threat.position) < 15:
                    agent.move_away_from(threat.position, speed_multiplier=1.5)
                else:
                    food = self.find_nearest_food_optimized(agent)
                    if food:
                        if agent.position.distance_to(food.position) < 2.0:
                            agent.energy = min(agent.max_energy, agent.energy + food.energy_value)
                            food.is_available = False
                            food.current_regen = 0
                        else:
                            agent.move_towards(food.position)
                    else:
                        agent.random_move()
            
            else:  # CARNIVORE
                # Optimized carnivore behavior
                prey = None
                # Find nearest herbivore
                nearby_agents = self.find_nearby_agents_optimized(agent, 30.0)
                herbivores = [a for a in nearby_agents if a.species_type == SpeciesType.HERBIVORE]
                if herbivores:
                    prey = min(herbivores, key=lambda h: agent.position.distance_to(h.position))
                
                if prey:
                    if self.attempt_hunt(agent, prey):
                        pass  # Hunt successful
                    else:
                        agent.move_towards(prey.position, speed_multiplier=1.2)
                else:
                    agent.random_move()
            
            # Reproduction
            if agent.can_reproduce():
                offspring = self.create_offspring_optimized(agent)
                if offspring:
                    new_agents.append(offspring)
            
            # Update agent state
            agent.update()
            self.keep_agent_in_bounds(agent)
        
        # Remove dead agents
        for agent in agents_to_remove:
            self._remove_agent_optimized(agent)
        
        # Add new agents
        for new_agent in new_agents:
            self._add_agent_optimized(new_agent)
        
        # Food regeneration (less frequent updates for performance)
        if self.step_counter % self.config.update_frequency_food == 0:
            for food in self.food_sources:
                if not food.is_available:
                    food.current_regen += self.config.update_frequency_food
                    if food.current_regen >= food.regeneration_time:
                        food.is_available = True
        
        # Performance tracking
        step_time = time.time() - step_start_time
        self.performance_stats['total_steps'] += 1
        self.performance_stats['total_step_time'] += step_time
        
        if self.config.enable_performance_profiling and self.step_counter % 100 == 0:
            self._log_performance_stats()
    
    def create_offspring_optimized(self, parent: Agent) -> Optional[Agent]:
        """Create offspring using object pool if available"""
        # Use object pool if enabled
        if self.config.enable_object_pooling:
            if parent.species_type == SpeciesType.HERBIVORE:
                offspring = self.herbivore_pool.get_agent()
            else:
                offspring = self.carnivore_pool.get_agent()
            
            self.performance_stats['memory_pool_hits'] += 1
        else:
            # Create new agent
            offspring_position = Position(parent.position.x + np.random.uniform(-5, 5),
                                        parent.position.y + np.random.uniform(-5, 5))
            agent_id = int(time.time() * 1000000) % 1000000  # Simple ID generation
            offspring = Agent(parent.species_type, offspring_position, agent_id)
        
        # Initialize offspring
        offspring.energy = parent.energy * 0.6  # Parent gives energy to offspring
        parent.energy *= 0.4  # Parent loses energy from reproduction
        offspring.generation = getattr(parent, 'generation', 0) + 1
        
        return offspring
    
    def _add_agent_optimized(self, agent: Agent):
        """Add agent with optimizations"""
        self.agents.append(agent)
        
        # Add to spatial index
        if self.config.use_spatial_indexing and hasattr(self, 'spatial_system'):
            agent.spatial_id = self.spatial_system.add_agent(agent)
        
        # Add to vectorized arrays
        if self.config.enable_vectorized_updates:
            self._add_agent_to_arrays(agent)
    
    def _remove_agent_optimized(self, agent: Agent):
        """Remove agent with optimizations"""
        if agent in self.agents:
            self.agents.remove(agent)
        
        # Remove from spatial index
        if (self.config.use_spatial_indexing and hasattr(self, 'spatial_system') and
            hasattr(agent, 'spatial_id')):
            self.spatial_system.remove_agent(agent.spatial_id)
        
        # Remove from vectorized arrays
        if self.config.enable_vectorized_updates:
            self._remove_agent_from_arrays(agent)
        
        # Return to object pool
        if self.config.enable_object_pooling:
            if agent.species_type == SpeciesType.HERBIVORE:
                self.herbivore_pool.return_agent(agent)
            else:
                self.carnivore_pool.return_agent(agent)
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        avg_step_time = (self.performance_stats['total_step_time'] / 
                        self.performance_stats['total_steps'] * 1000)
        
        print(f"🚀 Performance Stats (Step {self.step_counter}):")
        print(f"   Average step time: {avg_step_time:.2f}ms")
        print(f"   Steps per second: {1000/avg_step_time:.1f}")
        print(f"   Spatial queries: {self.performance_stats['spatial_queries']}")
        print(f"   Active agents: {len(self.agents)}")
        
        if hasattr(self, 'spatial_system'):
            spatial_stats = self.spatial_system.get_spatial_performance_stats()
            print(f"   Spatial efficiency: {spatial_stats['agents']['grid_utilization']:.1%}")
    
    def get_optimization_stats(self) -> Dict[str, any]:
        """Get comprehensive optimization statistics"""
        stats = {
            'performance': self.performance_stats.copy(),
            'config': {
                'spatial_indexing': self.config.use_spatial_indexing,
                'vectorized_updates': self.config.enable_vectorized_updates,
                'object_pooling': self.config.enable_object_pooling
            }
        }
        
        # Add spatial stats if available
        if hasattr(self, 'spatial_system'):
            stats['spatial'] = self.spatial_system.get_spatial_performance_stats()
        
        # Add pool stats if available
        if self.config.enable_object_pooling:
            stats['pools'] = {
                'herbivore': self.herbivore_pool.get_stats(),
                'carnivore': self.carnivore_pool.get_stats()
            }
        
        return stats
    
    # Override the step method to use optimized version
    def step(self):
        """Use optimized step implementation"""
        self.step_optimized()

# Factory function for creating optimized environments
def create_optimized_environment(width: int = 100, height: int = 100, 
                                performance_level: str = "high") -> HighPerformanceEcosystem:
    """Create an optimized environment with preset performance configurations"""
    
    configs = {
        "low": PerformanceConfig(
            use_spatial_indexing=False,
            enable_vectorized_updates=False,
            enable_object_pooling=False,
            enable_performance_profiling=True
        ),
        "medium": PerformanceConfig(
            use_spatial_indexing=True,
            spatial_cell_size=30.0,
            enable_vectorized_updates=False,
            enable_object_pooling=True,
            enable_performance_profiling=True
        ),
        "high": PerformanceConfig(
            use_spatial_indexing=True,
            spatial_cell_size=25.0,
            enable_vectorized_updates=True,
            enable_object_pooling=True,
            max_neighbor_distance=40.0,
            enable_performance_profiling=True
        ),
        "maximum": PerformanceConfig(
            use_spatial_indexing=True,
            spatial_cell_size=20.0,
            enable_vectorized_updates=True,
            enable_object_pooling=True,
            max_neighbor_distance=35.0,
            update_frequency_food=2,  # Less frequent food updates
            enable_performance_profiling=False  # Disable for maximum speed
        )
    }
    
    config = configs.get(performance_level, configs["high"])
    return HighPerformanceEcosystem(width, height, config)
