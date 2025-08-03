#!/usr/bin/env python3
"""
Analyze and Fix Carnivore Reproduction Issues
===========================================

This script analyzes the current carnivore behavior issues:
1. Carnivores continue reproducing after killing all herbivores
2. Energy costs may not be properly applied when starving
3. Speed/vision parameters may need adjustment

Phase 2 Enhanced Analysis
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

def analyze_carnivore_behavior():
    """Analyze current carnivore behavior issues"""
    print("🔬 ANALYZING CARNIVORE BEHAVIOR ISSUES")
    print("=" * 50)
    
    # Create test environment
    env = Phase2NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Get initial counts
    initial_herbivores = [a for a in env.agents if a.species_type == SpeciesType.HERBIVORE]
    initial_carnivores = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE]
    
    print(f"🦌 Initial Herbivores: {len(initial_herbivores)}")
    print(f"🐺 Initial Carnivores: {len(initial_carnivores)}")
    print(f"🌱 Food Sources: {len(env.food_sources)}")
    
    # Analyze carnivore stats
    print("\n📊 INITIAL CARNIVORE ANALYSIS:")
    for i, carnivore in enumerate(initial_carnivores[:3]):  # Check first 3
        print(f"Carnivore {i+1}:")
        print(f"  Energy: {carnivore.energy:.1f}/{carnivore.max_energy}")
        print(f"  Speed: {carnivore.speed}")
        print(f"  Vision Range: {carnivore.vision_range}")
        print(f"  Reproduction Threshold: {carnivore.config.reproduction_cost}")
        print(f"  Can Reproduce: {carnivore.can_reproduce()}")
        print(f"  Steps Since Fed: {carnivore.steps_since_fed}")
        
    # Run simulation until herbivores are extinct
    step = 0
    herbivore_extinction_step = None
    
    print("\n🏃 RUNNING SIMULATION TO HERBIVORE EXTINCTION...")
    
    while step < 1000:  # Max 1000 steps
        step += 1
        
        # Count populations before step
        herbivores_before = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        carnivores_before = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive])
        
        env.step()
        
        # Count populations after step
        herbivores_after = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        carnivores_after = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive])
        
        # Check for herbivore extinction
        if herbivores_after == 0 and herbivore_extinction_step is None:
            herbivore_extinction_step = step
            print(f"💀 HERBIVORE EXTINCTION AT STEP {step}")
            print(f"   Remaining carnivores: {carnivores_after}")
            
        # Monitor carnivore reproduction after extinction
        if herbivore_extinction_step and step > herbivore_extinction_step:
            if carnivores_after > carnivores_before:
                print(f"❌ PROBLEM: Carnivore reproduction at step {step} (after extinction)")
                print(f"   Carnivores: {carnivores_before} → {carnivores_after}")
                
                # Analyze the reproducing carnivores
                for carnivore in env.agents:
                    if (carnivore.species_type == SpeciesType.CARNIVORE and 
                        carnivore.is_alive and carnivore.reproduction_cooldown > 0):
                        print(f"   Recent reproducer: Energy {carnivore.energy:.1f}, "
                              f"Steps since fed: {carnivore.steps_since_fed}")
        
        # Print periodic updates
        if step % 100 == 0:
            print(f"Step {step}: 🦌 {herbivores_after} herbivores, 🐺 {carnivores_after} carnivores")
        
        # Stop if all agents are dead
        if len([a for a in env.agents if a.is_alive]) == 0:
            print(f"💀 TOTAL ECOSYSTEM COLLAPSE AT STEP {step}")
            break
    
    return analyze_post_extinction_behavior(env, herbivore_extinction_step)

def analyze_post_extinction_behavior(env, extinction_step):
    """Analyze carnivore behavior after herbivore extinction"""
    print(f"\n🔍 POST-EXTINCTION ANALYSIS (after step {extinction_step})")
    print("=" * 50)
    
    carnivores = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive]
    
    if not carnivores:
        print("❌ No surviving carnivores to analyze")
        return None
    
    print(f"🐺 Surviving carnivores: {len(carnivores)}")
    
    # Analyze each carnivore's state
    reproductive_carnivores = 0
    starving_carnivores = 0
    
    for i, carnivore in enumerate(carnivores):
        print(f"\nCarnivore {i+1}:")
        print(f"  Energy: {carnivore.energy:.1f}/{carnivore.max_energy}")
        print(f"  Age: {carnivore.age}")
        print(f"  Steps since fed: {carnivore.steps_since_fed}")
        print(f"  Reproduction cooldown: {carnivore.reproduction_cooldown}")
        print(f"  Can reproduce: {carnivore.can_reproduce()}")
        
        if carnivore.can_reproduce():
            reproductive_carnivores += 1
            print(f"  ❌ PROBLEM: Can still reproduce without food!")
        
        if carnivore.steps_since_fed > 40:
            starving_carnivores += 1
            print(f"  🔥 Starving (should have high energy cost)")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Reproductive carnivores: {reproductive_carnivores}/{len(carnivores)}")
    print(f"  Starving carnivores: {starving_carnivores}/{len(carnivores)}")
    
    return {
        'total_carnivores': len(carnivores),
        'reproductive_carnivores': reproductive_carnivores,
        'starving_carnivores': starving_carnivores,
        'extinction_step': extinction_step
    }

def propose_fixes():
    """Propose fixes for the carnivore issues"""
    print("\n🛠️ PROPOSED FIXES")
    print("=" * 50)
    
    print("1. 🚫 STRICTER CARNIVORE REPRODUCTION:")
    print("   - Increase reproduction requirements")
    print("   - Add minimum prey availability check")
    print("   - Increase energy cost when starving")
    
    print("\n2. ⚡ SPEED/VISION ADJUSTMENTS:")
    print("   - Reduce carnivore speed to prevent overhunting")
    print("   - Reduce carnivore vision range for balance")
    print("   - Increase herbivore escape abilities")
    
    print("\n3. 🔥 ENHANCED STARVATION MECHANICS:")
    print("   - Exponential energy decay when no food available")
    print("   - Block reproduction completely when starving")
    print("   - Add desperation behaviors")
    
    print("\n4. 🌍 ENVIRONMENTAL BALANCE:")
    print("   - Increase food regeneration for herbivores")
    print("   - Add carnivore-specific energy penalties")
    print("   - Implement population-based reproduction limits")

def test_proposed_fixes():
    """Test the proposed fixes"""
    print("\n🧪 TESTING PROPOSED FIXES")
    print("=" * 50)
    
    # Create modified configuration
    config = EvolutionaryAgentConfig()
    # Stricter reproduction requirements
    config.reproduction_cost = 70.0  # Increased from 50
    config.carnivore_energy_cost = 2.0  # Increased starvation cost
    config.reproduction_cooldown = 40  # Longer cooldown
    
    network_config = EvolutionaryNetworkConfig()
    
    # Create test carnivore
    test_carnivore = EvolutionaryNeuralAgent(
        SpeciesType.CARNIVORE, 
        Position(50, 50), 
        1,
        config=config,
        network_config=network_config
    )
    
    # Reduce speed and vision for balance
    test_carnivore.speed = 2.0  # Reduced from 2.5
    test_carnivore.vision_range = 25.0  # Reduced from 30.0
    
    print(f"🐺 Modified Carnivore Stats:")
    print(f"  Speed: {test_carnivore.speed} (was 2.5)")
    print(f"  Vision: {test_carnivore.vision_range} (was 30.0)")
    print(f"  Reproduction cost: {config.reproduction_cost} (was 50.0)")
    print(f"  Starvation cost: {config.carnivore_energy_cost} (was 1.0)")
    
    # Test starvation behavior
    print(f"\n🔥 Testing starvation mechanics:")
    test_carnivore.steps_since_fed = 50  # Simulate long starvation
    
    original_energy = test_carnivore.energy
    test_carnivore.update()  # Trigger starvation penalty
    
    print(f"  Energy before starvation update: {original_energy:.1f}")
    print(f"  Energy after starvation update: {test_carnivore.energy:.1f}")
    print(f"  Energy lost: {original_energy - test_carnivore.energy:.1f}")
    print(f"  Can reproduce when starving: {test_carnivore.can_reproduce()}")

if __name__ == "__main__":
    try:
        # Run analysis
        results = analyze_carnivore_behavior()
        
        # Propose fixes
        propose_fixes()
        
        # Test fixes
        test_proposed_fixes()
        
        print("\n✅ ANALYSIS COMPLETE!")
        print("Check the output above for identified issues and proposed solutions.")
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
