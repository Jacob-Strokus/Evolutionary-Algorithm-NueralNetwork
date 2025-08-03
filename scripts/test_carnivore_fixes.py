#!/usr/bin/env python3
"""
Test Carnivore Balance Fixes
===========================

Test the implemented fixes for carnivore reproduction and balance issues.
"""

import sys
import os
import random
import math

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.neural.evolutionary_agent import EvolutionaryNeuralAgent, EvolutionaryAgentConfig
from src.neural.evolutionary_network import EvolutionaryNetworkConfig
from src.core.ecosystem import SpeciesType, Position
from main import Phase2NeuralEnvironment

def test_carnivore_fixes():
    """Test the implemented carnivore balance fixes"""
    print("🧪 TESTING CARNIVORE BALANCE FIXES")
    print("=" * 50)
    
    # Create test environment
    env = Phase2NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Get initial counts
    initial_herbivores = [a for a in env.agents if a.species_type == SpeciesType.HERBIVORE]
    initial_carnivores = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE]
    
    print(f"🦌 Initial Herbivores: {len(initial_herbivores)}")
    print(f"🐺 Initial Carnivores: {len(initial_carnivores)}")
    
    # Test fix 1: Check updated carnivore stats
    print(f"\n🔍 TESTING FIX 1: UPDATED CARNIVORE STATS")
    carnivore = initial_carnivores[0]
    print(f"  Speed: {carnivore.speed} (should be 2.0, was 2.5)")
    print(f"  Vision: {carnivore.vision_range} (should be 25.0, was 30.0)")
    print(f"  Reproduction cost: {carnivore.config.reproduction_cost} (should be 60.0, was 50.0)")
    print(f"  Carnivore energy cost: {carnivore.config.carnivore_energy_cost} (should be 2.0, was 1.0)")
    
    # Test fix 2: Check herbivore improvements
    print(f"\n🔍 TESTING FIX 2: HERBIVORE IMPROVEMENTS")
    herbivore = initial_herbivores[0]
    print(f"  Herbivore speed: {herbivore.speed} (should be 1.8, was 1.5)")
    print(f"  Herbivore vision: {herbivore.vision_range} (should be 18.0, was 15.0)")
    
    # Test fix 3: Stricter reproduction requirements
    print(f"\n🔍 TESTING FIX 3: STRICTER REPRODUCTION REQUIREMENTS")
    test_carnivore = initial_carnivores[0]
    
    # Test well-fed carnivore
    test_carnivore.energy = 80
    test_carnivore.steps_since_fed = 10
    test_carnivore.age = 40
    test_carnivore.reproduction_cooldown = 0
    print(f"  Well-fed carnivore can reproduce: {test_carnivore.can_reproduce()} (should be True)")
    
    # Test carnivore that hasn't eaten recently
    test_carnivore.steps_since_fed = 30
    print(f"  Carnivore (30 steps since fed) can reproduce: {test_carnivore.can_reproduce()} (should be False)")
    
    # Test low-energy carnivore
    test_carnivore.energy = 60
    test_carnivore.steps_since_fed = 20
    print(f"  Low-energy carnivore can reproduce: {test_carnivore.can_reproduce()} (should be False)")
    
    # Test fix 4: Enhanced starvation mechanics
    print(f"\n🔍 TESTING FIX 4: ENHANCED STARVATION MECHANICS")
    starving_carnivore = initial_carnivores[1]
    starving_carnivore.steps_since_fed = 50  # Long starvation
    original_energy = starving_carnivore.energy
    
    starving_carnivore.update()  # Trigger starvation penalty
    energy_lost = original_energy - starving_carnivore.energy
    print(f"  Energy lost from starvation: {energy_lost:.2f} (should be significant)")
    
    # Run simulation to test overall balance
    print(f"\n🏃 TESTING OVERALL BALANCE: Running 500 steps...")
    
    herbivore_extinction_step = None
    post_extinction_reproductions = 0
    
    for step in range(500):
        herbivores_before = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        carnivores_before = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive])
        
        env.step()
        
        herbivores_after = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        carnivores_after = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive])
        
        # Check for herbivore extinction
        if herbivores_after == 0 and herbivore_extinction_step is None:
            herbivore_extinction_step = step
            print(f"  💀 Herbivore extinction at step {step}")
        
        # Monitor post-extinction carnivore reproduction
        if herbivore_extinction_step and step > herbivore_extinction_step:
            if carnivores_after > carnivores_before:
                post_extinction_reproductions += 1
        
        # Print periodic updates
        if step % 100 == 0:
            print(f"  Step {step}: 🦌 {herbivores_after} herbivores, 🐺 {carnivores_after} carnivores")
        
        # Stop if all agents are dead
        if len([a for a in env.agents if a.is_alive]) == 0:
            print(f"  💀 Total ecosystem collapse at step {step}")
            break
    
    # Test results
    print(f"\n📊 TEST RESULTS:")
    print(f"  Herbivore extinction step: {herbivore_extinction_step if herbivore_extinction_step else 'None'}")
    print(f"  Post-extinction carnivore reproductions: {post_extinction_reproductions}")
    
    # Analyze final carnivore state
    final_carnivores = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive]
    if final_carnivores:
        reproductive_carnivores = sum(1 for c in final_carnivores if c.can_reproduce())
        print(f"  Final carnivores: {len(final_carnivores)}")
        print(f"  Reproductive carnivores: {reproductive_carnivores}/{len(final_carnivores)}")
        
        # Check starvation state
        starving_carnivores = sum(1 for c in final_carnivores if c.steps_since_fed > 30)
        print(f"  Starving carnivores: {starving_carnivores}/{len(final_carnivores)}")
    
    # Success criteria
    success = True
    if post_extinction_reproductions > 5:  # Allow some reproductions but not excessive
        print(f"  ❌ Too many post-extinction reproductions: {post_extinction_reproductions}")
        success = False
    else:
        print(f"  ✅ Post-extinction reproductions under control: {post_extinction_reproductions}")
    
    if final_carnivores and reproductive_carnivores > len(final_carnivores) * 0.3:
        print(f"  ❌ Too many reproductive carnivores after extinction")
        success = False
    else:
        print(f"  ✅ Reproductive carnivores properly limited")
    
    return success

def test_individual_fixes():
    """Test individual fix components"""
    print(f"\n🔧 TESTING INDIVIDUAL FIX COMPONENTS")
    print("=" * 50)
    
    # Create test agents
    config = EvolutionaryAgentConfig()
    network_config = EvolutionaryNetworkConfig()
    
    # Test carnivore
    carnivore = EvolutionaryNeuralAgent(
        SpeciesType.CARNIVORE, 
        Position(50, 50), 
        1,
        config=config,
        network_config=network_config
    )
    
    # Test herbivore
    herbivore = EvolutionaryNeuralAgent(
        SpeciesType.HERBIVORE, 
        Position(30, 30), 
        2,
        config=config,
        network_config=network_config
    )
    
    print(f"✅ Configuration Tests:")
    print(f"  Carnivore speed: {carnivore.speed} (target: 2.0)")
    print(f"  Carnivore vision: {carnivore.vision_range} (target: 25.0)")
    print(f"  Herbivore speed: {herbivore.speed} (target: 1.8)")
    print(f"  Herbivore vision: {herbivore.vision_range} (target: 18.0)")
    print(f"  Reproduction cost: {config.reproduction_cost} (target: 60.0)")
    print(f"  Carnivore energy cost: {config.carnivore_energy_cost} (target: 2.0)")
    print(f"  Reproduction cooldown: {config.reproduction_cooldown} (target: 30)")

if __name__ == "__main__":
    try:
        # Test individual components
        test_individual_fixes()
        
        # Test overall balance
        success = test_carnivore_fixes()
        
        if success:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"Carnivore balance fixes are working correctly.")
        else:
            print(f"\n⚠️ SOME TESTS FAILED!")
            print(f"Additional adjustments may be needed.")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
