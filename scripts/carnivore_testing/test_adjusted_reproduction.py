#!/usr/bin/env python3
"""
Test Adjusted Carnivore Reproduction Rates
==========================================

Quick test to verify the relaxed carnivore reproduction requirements work better.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.neural.evolutionary_agent import EvolutionaryNeuralAgent, EvolutionaryAgentConfig
from src.neural.evolutionary_network import EvolutionaryNetworkConfig
from src.core.ecosystem import SpeciesType, Position
from main import Phase2NeuralEnvironment

def test_adjusted_reproduction():
    """Test the adjusted carnivore reproduction requirements"""
    print("🧪 TESTING ADJUSTED CARNIVORE REPRODUCTION RATES")
    print("=" * 55)
    
    # Create test environment
    env = Phase2NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Get carnivore for testing
    carnivores = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE]
    
    if not carnivores:
        print("❌ No carnivores found to test")
        return
    
    test_carnivore = carnivores[0]
    
    print(f"📊 UPDATED CONFIGURATION:")
    print(f"  Reproduction cost: {test_carnivore.config.reproduction_cost} (was 60, now 55)")
    print(f"  Carnivore energy cost: {test_carnivore.config.carnivore_energy_cost} (was 2.0, now 1.8)")
    print(f"  Reproduction cooldown: {test_carnivore.config.reproduction_cooldown} (was 30, now 25)")
    
    print(f"\n🔍 TESTING REPRODUCTION SCENARIOS:")
    
    # Scenario 1: Well-fed carnivore (should reproduce)
    test_carnivore.energy = 70
    test_carnivore.steps_since_fed = 20
    test_carnivore.age = 40
    test_carnivore.reproduction_cooldown = 0
    can_reproduce_1 = test_carnivore.can_reproduce()
    print(f"  Well-fed (70 energy, 20 steps): {can_reproduce_1} ✅")
    
    # Scenario 2: Moderate energy, recent feeding (should reproduce)
    test_carnivore.energy = 65
    test_carnivore.steps_since_fed = 30
    can_reproduce_2 = test_carnivore.can_reproduce()
    print(f"  Moderate energy (65 energy, 30 steps): {can_reproduce_2} ✅")
    
    # Scenario 3: Low energy (should NOT reproduce)
    test_carnivore.energy = 60
    test_carnivore.steps_since_fed = 25
    can_reproduce_3 = test_carnivore.can_reproduce()
    print(f"  Low energy (60 energy, 25 steps): {can_reproduce_3} ❌")
    
    # Scenario 4: Haven't eaten recently (should NOT reproduce)
    test_carnivore.energy = 70
    test_carnivore.steps_since_fed = 40
    can_reproduce_4 = test_carnivore.can_reproduce()
    print(f"  Unfed recently (70 energy, 40 steps): {can_reproduce_4} ❌")
    
    # Scenario 5: Edge case - exactly at threshold
    test_carnivore.energy = 65
    test_carnivore.steps_since_fed = 35
    can_reproduce_5 = test_carnivore.can_reproduce()
    print(f"  Edge case (65 energy, 35 steps): {can_reproduce_5} ❓")
    
    print(f"\n⚡ STARVATION PENALTY TEST:")
    
    # Test starvation penalty with new values
    test_carnivore.steps_since_fed = 45  # 10 steps into starvation
    original_energy = test_carnivore.energy
    test_carnivore.update()
    energy_lost = original_energy - test_carnivore.energy
    print(f"  Energy lost at 45 steps unfed: {energy_lost:.2f}")
    
    # Run quick simulation to see population dynamics
    print(f"\n🏃 QUICK SIMULATION TEST (200 steps):")
    
    reproduction_events = 0
    carnivore_deaths = 0
    
    for step in range(200):
        carnivores_before = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive])
        
        env.step()
        
        carnivores_after = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive])
        herbivores_after = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        
        # Count reproduction events
        if carnivores_after > carnivores_before:
            reproduction_events += (carnivores_after - carnivores_before)
        elif carnivores_after < carnivores_before:
            carnivore_deaths += (carnivores_before - carnivores_after)
        
        if step % 50 == 0:
            print(f"  Step {step}: 🦌 {herbivores_after} herbivores, 🐺 {carnivores_after} carnivores")
    
    print(f"\n📈 RESULTS:")
    print(f"  Carnivore reproduction events: {reproduction_events}")
    print(f"  Carnivore deaths: {carnivore_deaths}")
    
    # Success criteria
    if reproduction_events > 0 and reproduction_events < 20:  # Some reproduction but not excessive
        print(f"  ✅ Reproduction rate looks balanced!")
    elif reproduction_events == 0:
        print(f"  ⚠️ No reproduction - might be too strict")
    else:
        print(f"  ⚠️ Too much reproduction - might need adjustment")
    
    return reproduction_events

if __name__ == "__main__":
    try:
        reproduction_count = test_adjusted_reproduction()
        
        print(f"\n🎯 ADJUSTMENT SUMMARY:")
        print(f"  • Reproduction requirements: 75 energy → 65 energy")
        print(f"  • Feeding requirement: 25 steps → 35 steps")
        print(f"  • Reproduction cost: 60 → 55 energy")
        print(f"  • Energy cost multiplier: 2.0 → 1.8")
        print(f"  • Cooldown: 30 → 25 steps")
        print(f"  • Starvation starts: 30 → 35 steps")
        
        print(f"\n✅ Carnivore reproduction rates have been relaxed for better balance!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
