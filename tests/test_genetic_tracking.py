#!/usr/bin/env python3
"""
Test script for genetic operations tracking
Tests mutations and crossovers display
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment
import time

def test_genetic_tracking():
    """Test the genetic operations tracking system"""
    print("🧬 Testing Genetic Operations Tracking")
    print("=" * 50)
    
    # Create environment
    env = NeuralEnvironment(width=200, height=200, use_neural_agents=True)
    print(f"✅ Environment created with {len(env.agents)} neural agents")
    
    # Run simulation for a few steps to generate genetic events
    print("\n🔄 Running simulation to generate genetic operations...")
    
    for step in range(20):
        env.step()
        
        # Get current statistics
        stats = env.get_neural_stats()
        genetic_stats = env.genetic_algorithm.get_genetic_stats()
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  Agents: H={stats.get('herbivore_count', 0)}, C={stats.get('carnivore_count', 0)}")
            print(f"  Recent Mutations: {genetic_stats['recent_mutations']}")
            print(f"  Recent Crossovers: {genetic_stats['recent_crossovers']}")
            print(f"  Total Mutations: {genetic_stats['total_mutations']}")
            print(f"  Total Crossovers: {genetic_stats['total_crossovers']}")
            print(f"  Mutation Rate/min: {genetic_stats['mutation_rate']}")
            print(f"  Crossover Rate/min: {genetic_stats['crossover_rate']}")
            
            # Show recent events
            recent_events = genetic_stats['recent_events']
            if recent_events:
                print(f"  Recent Events: {len(recent_events)} events")
                for event in recent_events[-3:]:  # Show last 3 events
                    event_type = event.get('type', 'unknown')
                    species = event.get('parent_species', 'unknown')
                    fitness = event.get('parent_fitness', 0)
                    print(f"    - {event_type}: {species} (fitness: {fitness:.1f})")
        
        # Reset counters every few steps
        if step % 10 == 9:
            env.genetic_algorithm.reset_recent_counters()
            print("  🔄 Reset recent counters")
        
        time.sleep(0.1)  # Small delay to simulate real-time
    
    print("\n✅ Genetic tracking test completed!")
    print(f"📊 Final Statistics:")
    
    final_stats = env.genetic_algorithm.get_genetic_stats()
    print(f"  Total Mutations: {final_stats['total_mutations']}")
    print(f"  Total Crossovers: {final_stats['total_crossovers']}")
    print(f"  Recent Events: {len(final_stats['recent_events'])}")
    
    return True

if __name__ == "__main__":
    try:
        test_genetic_tracking()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
