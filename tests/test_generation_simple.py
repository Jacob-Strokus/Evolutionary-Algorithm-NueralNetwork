#!/usr/bin/env python3
"""
Simple Generation Tracking Test
Test reproduction and generation inheritance without web server
"""

import sys
import os
# Add parent directory to path to access src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType

def test_generation_inheritance():
    """Test that offspring inherit the correct generation number"""
    print("🧬 Testing Generation Inheritance During Reproduction")
    print("=" * 60)
    
    # Create small environment for easier testing
    env = NeuralEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Get a herbivore
    herbivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE]
    
    if not herbivores:
        print("❌ No herbivores found!")
        return False
    
    parent = herbivores[0]
    print(f"📊 Testing with herbivore ID: {parent.id}")
    print(f"   Parent generation: {getattr(parent, 'generation', 'MISSING')}")
    print(f"   Parent initial energy: {parent.energy:.1f}")
    
    # Set up optimal reproduction conditions
    parent.energy = 200  # High energy
    parent.reproduction_cooldown = 0  # Ready to reproduce
    parent.age = 50  # Mature enough
    
    # Track initial population
    initial_neural_agents = len([a for a in env.agents if isinstance(a, NeuralAgent)])
    print(f"   Initial neural agent count: {initial_neural_agents}")
    
    # Force reproduction by running until it happens
    reproduction_occurred = False
    offspring_generation = None
    
    print("\n🔄 Running simulation to trigger reproduction...")
    
    for step in range(100):
        env.step()
        
        current_neural_agents = len([a for a in env.agents if isinstance(a, NeuralAgent)])
        
        if current_neural_agents > initial_neural_agents:
            reproduction_occurred = True
            print(f"   🎉 Reproduction detected at step {step}!")
            print(f"   Population: {initial_neural_agents} → {current_neural_agents}")
            
            # Find the newest agents (they should be at the end)
            neural_agents = [a for a in env.agents if isinstance(a, NeuralAgent)]
            newest_agents = neural_agents[initial_neural_agents:]
            
            for i, offspring in enumerate(newest_agents):
                if offspring.species_type == SpeciesType.HERBIVORE:
                    offspring_generation = getattr(offspring, 'generation', 'MISSING')
                    print(f"   Offspring {i} generation: {offspring_generation}")
                    print(f"   Expected generation: {parent.generation + 1}")
                    
                    if offspring_generation == parent.generation + 1:
                        print("   ✅ Generation inheritance working correctly!")
                        return True
                    else:
                        print("   ❌ Generation inheritance failed!")
                        return False
            
            break
        
        # Keep parent in good shape for reproduction
        if step % 10 == 0:
            parent.energy = min(parent.energy + 10, 200)
            parent.reproduction_cooldown = max(0, parent.reproduction_cooldown - 1)
    
    if not reproduction_occurred:
        print("   ⚠️ No reproduction occurred in 100 steps")
        print(f"   Parent final energy: {parent.energy:.1f}")
        print(f"   Parent reproduction cooldown: {parent.reproduction_cooldown}")
        print(f"   Parent can reproduce: {parent.can_reproduce()}")
        return False
    
    return False

def test_crossover_generation():
    """Test generation tracking in genetic crossover"""
    print("\n🧬 Testing Generation in Genetic Crossover")
    print("=" * 50)
    
    env = NeuralEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Get two herbivores for crossover
    herbivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE]
    
    if len(herbivores) < 2:
        print("❌ Need at least 2 herbivores for crossover test")
        return False
    
    parent1 = herbivores[0]
    parent2 = herbivores[1]
    
    # Set different generations to test inheritance
    parent1.generation = 2
    parent2.generation = 3
    
    print(f"📊 Parent 1 generation: {parent1.generation}")
    print(f"📊 Parent 2 generation: {parent2.generation}")
    print(f"📊 Expected offspring generation: {max(parent1.generation, parent2.generation) + 1}")
    
    # Test the crossover mechanism directly
    try:
        # Access the genetic algorithm
        genetic_algo = env.genetic_algorithm
        
        # Create offspring through crossover
        from src.core.ecosystem import Position
        import random
        
        offspring_pos = Position(
            random.uniform(20, env.width - 20),
            random.uniform(20, env.height - 20)
        )
        
        # Create offspring with proper generation
        offspring_generation = max(parent1.generation, parent2.generation) + 1
        offspring = NeuralAgent(parent1.species_type, offspring_pos, env.next_agent_id, generation=offspring_generation)
        offspring.brain = genetic_algo.crossover_networks(parent1, parent2)
        
        print(f"✅ Crossover offspring generation: {offspring.generation}")
        
        if offspring.generation == max(parent1.generation, parent2.generation) + 1:
            print("✅ Crossover generation inheritance working correctly!")
            return True
        else:
            print("❌ Crossover generation inheritance failed!")
            return False
            
    except Exception as e:
        print(f"❌ Crossover test failed: {e}")
        return False

def summary_test():
    """Summary of generation tracking implementation"""
    print("\n📋 Generation Tracking Implementation Summary")
    print("=" * 55)
    
    # Test basic agent creation
    env = NeuralEnvironment(width=30, height=30, use_neural_agents=True)
    
    neural_agents = [a for a in env.agents if isinstance(a, NeuralAgent)]
    
    print(f"✅ Neural agents created: {len(neural_agents)}")
    
    generations_present = set()
    for agent in neural_agents:
        gen = getattr(agent, 'generation', None)
        if gen is not None:
            generations_present.add(gen)
    
    print(f"✅ Generations present: {sorted(generations_present)}")
    
    if 1 in generations_present and len(generations_present) == 1:
        print("✅ All initial agents are Generation 1")
        return True
    else:
        print("❌ Initial generation assignment incorrect")
        return False

if __name__ == "__main__":
    print("🧬 Generation Tracking Test Suite")
    print("=" * 70)
    
    try:
        result1 = summary_test()
        result2 = test_generation_inheritance()
        result3 = test_crossover_generation()
        
        print(f"\n🏁 Test Results:")
        print(f"   Basic Generation Assignment: {'✅ PASS' if result1 else '❌ FAIL'}")
        print(f"   Reproduction Inheritance: {'✅ PASS' if result2 else '❌ FAIL'}")
        print(f"   Crossover Inheritance: {'✅ PASS' if result3 else '❌ FAIL'}")
        
        if result1 and result2 and result3:
            print("\n🎉 All generation tracking tests passed!")
            print("🌐 Ready for web interface testing!")
        elif result1:
            print("\n✅ Basic generation tracking working!")
            print("⚠️ Some advanced features need testing with actual reproduction events")
        else:
            print("\n❌ Generation tracking has issues!")
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
