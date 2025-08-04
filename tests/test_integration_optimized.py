#!/usr/bin/env python3
"""
Integration tests for optimized ecosystem performance
Tests the integration of optimization features with the main simulation
"""

import sys
import os
import time
import unittest
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimization.high_performance_ecosystem import create_optimized_environment
from optimization.spatial_indexing import SpatialGrid
from core.ecosystem import Environment, SpeciesType
from neural.evolutionary_agent import EvolutionaryAgent

class TestOptimizedIntegration(unittest.TestCase):
    """Integration tests for optimized ecosystem"""
    
    def setUp(self):
        """Set up test environment"""
        self.width = 800
        self.height = 600
        
    def test_optimized_ecosystem_creation(self):
        """Test optimized ecosystem can be created with different performance levels"""
        for level in ["low", "medium", "high", "maximum"]:
            with self.subTest(performance_level=level):
                env = create_optimized_environment(
                    self.width, 
                    self.height, 
                    performance_level=level
                )
                self.assertIsNotNone(env)
                self.assertEqual(env.width, self.width)
                self.assertEqual(env.height, self.height)
                
    def test_spatial_indexing_integration(self):
        """Test spatial indexing works with ecosystem agents"""
        env = create_optimized_environment(self.width, self.height, "high")
        
        # Add some agents
        for i in range(20):
            x = np.random.randint(50, self.width - 50)
            y = np.random.randint(50, self.height - 50)
            agent = EvolutionaryAgent(SpeciesType.HERBIVORE, (x, y), f"test_{i}")
            env.add_agent(agent)
            
        # Test spatial queries work
        if hasattr(env, 'spatial_grid'):
            test_x, test_y = self.width // 2, self.height // 2
            nearby = env.spatial_grid.query_radius(test_x, test_y, 100)
            self.assertIsInstance(nearby, list)
            
    def test_performance_comparison(self):
        """Test optimized vs standard ecosystem performance"""
        # Create both environments
        standard_env = Environment(self.width, self.height)
        optimized_env = create_optimized_environment(self.width, self.height, "high")
        
        # Add agents to both
        for i in range(30):
            x = np.random.randint(50, self.width - 50)
            y = np.random.randint(50, self.height - 50)
            
            agent1 = EvolutionaryAgent(SpeciesType.HERBIVORE, (x, y), f"std_{i}")
            agent2 = EvolutionaryAgent(SpeciesType.HERBIVORE, (x, y), f"opt_{i}")
            
            standard_env.add_agent(agent1)
            optimized_env.add_agent(agent2)
            
        # Time simulation steps
        steps = 100
        
        # Standard ecosystem timing
        start = time.time()
        for _ in range(steps):
            standard_env.step()
        standard_time = time.time() - start
        
        # Optimized ecosystem timing
        start = time.time()
        for _ in range(steps):
            optimized_env.step()
        optimized_time = time.time() - start
        
        # Optimized should be at least as fast (or within reasonable margin)
        self.assertLessEqual(optimized_time, standard_time * 1.2, 
                           f"Optimized: {optimized_time:.3f}s, Standard: {standard_time:.3f}s")
        
        print(f"\n📊 Performance Comparison:")
        print(f"   Standard: {standard_time:.3f}s ({steps/standard_time:.1f} steps/sec)")
        print(f"   Optimized: {optimized_time:.3f}s ({steps/optimized_time:.1f} steps/sec)")
        if optimized_time < standard_time:
            speedup = standard_time / optimized_time
            print(f"   Speedup: {speedup:.2f}x faster")
        
    def test_agent_pool_integration(self):
        """Test agent pool works correctly with ecosystem"""
        env = create_optimized_environment(self.width, self.height, "maximum")
        
        # Test agent pool if available
        if hasattr(env, 'agent_pools'):
            # Get agents from pool
            herbivore_pool = env.agent_pools.get(SpeciesType.HERBIVORE)
            if herbivore_pool:
                agent1 = herbivore_pool.get_agent()
                agent2 = herbivore_pool.get_agent()
                
                self.assertIsNotNone(agent1)
                self.assertIsNotNone(agent2)
                self.assertNotEqual(agent1, agent2)
                
                # Return agents to pool
                herbivore_pool.return_agent(agent1)
                herbivore_pool.return_agent(agent2)
                
    def test_large_population_stability(self):
        """Test ecosystem stability with larger populations"""
        env = create_optimized_environment(self.width, self.height, "high")
        
        # Add a larger population
        agent_count = 80
        for i in range(agent_count):
            x = np.random.randint(50, self.width - 50)
            y = np.random.randint(50, self.height - 50)
            species = SpeciesType.HERBIVORE if i % 3 != 0 else SpeciesType.CARNIVORE
            agent = EvolutionaryAgent(species, (x, y), f"large_{i}")
            env.add_agent(agent)
            
        # Run simulation for stability
        initial_count = len(env.agents)
        for step in range(50):
            env.step()
            # Ecosystem should remain stable (agents shouldn't all die immediately)
            if step == 25:  # Check midway
                self.assertGreater(len(env.agents), initial_count * 0.3, 
                                 "Too many agents died too quickly")
                
    def test_memory_efficiency(self):
        """Test memory usage doesn't grow excessively"""
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            self.skipTest("psutil not available for memory testing")
            
        env = create_optimized_environment(self.width, self.height, "high")
        
        # Baseline memory
        baseline_memory = process.memory_info().rss
        
        # Add agents and run simulation
        for i in range(60):
            x = np.random.randint(50, self.width - 50)
            y = np.random.randint(50, self.height - 50)
            agent = EvolutionaryAgent(SpeciesType.HERBIVORE, (x, y), f"mem_{i}")
            env.add_agent(agent)
            
        # Run steps
        for _ in range(100):
            env.step()
            
        # Check memory growth
        final_memory = process.memory_info().rss
        memory_growth = final_memory - baseline_memory
        
        # Memory growth should be reasonable (less than 50MB for this test)
        max_growth = 50 * 1024 * 1024  # 50MB
        self.assertLess(memory_growth, max_growth,
                       f"Memory grew by {memory_growth/1024/1024:.1f}MB, expected <50MB")
        
        print(f"\n💾 Memory Usage:")
        print(f"   Baseline: {baseline_memory/1024/1024:.1f}MB")
        print(f"   Final: {final_memory/1024/1024:.1f}MB")
        print(f"   Growth: {memory_growth/1024/1024:.1f}MB")

if __name__ == '__main__':
    unittest.main(verbosity=2)
