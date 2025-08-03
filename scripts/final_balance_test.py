#!/usr/bin/env python3
"""
Final Balance Test for Carnivore Reproduction
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main import Phase2NeuralEnvironment
from src.core.ecosystem import SpeciesType

def quick_balance_test():
    """Quick test of the final balanced parameters"""
    print("🎯 FINAL CARNIVORE REPRODUCTION BALANCE TEST")
    print("=" * 45)
    
    env = Phase2NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    reproduction_events = 0
    carnivore_deaths = 0
    herbivore_deaths = 0
    
    print("Running 300-step simulation...")
    
    for step in range(300):
        carnivores_before = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive])
        herbivores_before = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        
        env.step()
        
        carnivores_after = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive])
        herbivores_after = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        
        # Count events
        if carnivores_after > carnivores_before:
            reproduction_events += (carnivores_after - carnivores_before)
        elif carnivores_after < carnivores_before:
            carnivore_deaths += (carnivores_before - carnivores_after)
            
        if herbivores_after < herbivores_before:
            herbivore_deaths += (herbivores_before - herbivores_after)
        
        if step % 75 == 0:
            print(f"  Step {step:3d}: 🦌 {herbivores_after:2d} herbivores, 🐺 {carnivores_after:2d} carnivores")
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Carnivore reproductions: {reproduction_events}")
    print(f"  Carnivore deaths: {carnivore_deaths}")
    print(f"  Herbivore deaths: {herbivore_deaths}")
    
    # Balance assessment
    reproduction_rate = reproduction_events / 300 if reproduction_events > 0 else 0
    
    if 1 <= reproduction_events <= 10:
        print(f"  ✅ EXCELLENT: Balanced reproduction rate!")
    elif 11 <= reproduction_events <= 20:
        print(f"  ✅ GOOD: Moderate reproduction rate")
    elif reproduction_events == 0:
        print(f"  ⚠️ TOO STRICT: No reproduction occurred")
    else:
        print(f"  ⚠️ TOO PERMISSIVE: Excessive reproduction")
    
    return reproduction_events

if __name__ == "__main__":
    try:
        result = quick_balance_test()
        
        print(f"\n🎯 FINAL PARAMETERS:")
        print(f"  • Energy requirement: 68 (sweet spot)")
        print(f"  • Feeding requirement: 32 steps (sweet spot)")
        print(f"  • Reproduction cost: 55")
        print(f"  • Energy cost multiplier: 1.8")
        print(f"  • Reproduction cooldown: 25")
        print(f"  • Starvation threshold: 35 steps")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
