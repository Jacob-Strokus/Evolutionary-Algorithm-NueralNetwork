#!/usr/bin/env python3
"""
Script to analyze carnivore starvation dynamics and test fixes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import Phase2NeuralEnvironment
from src.core.ecosystem import SpeciesType

def analyze_carnivore_survival():
    """Analyze how long carnivores survive without food"""
    print("🔍 Analyzing Carnivore Starvation Dynamics")
    print("=" * 50)
    
    # Test current parameters
    env = Phase2NeuralEnvironment(100, 100)
    
    # Get a carnivore's stats
    carnivore_agents = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE]
    if carnivore_agents:
        carnivore = carnivore_agents[0]
        print(f"📊 Current Carnivore Parameters:")
        print(f"   • Starting Energy: {carnivore.energy}")
        print(f"   • Max Energy: {carnivore.max_energy}")
        print(f"   • Energy Decay: {carnivore.energy_decay} per step")
        print(f"   • Reproduction Threshold: {carnivore.reproduction_threshold}")
        print(f"   • Reproduction Cost: {carnivore.reproduction_cost}")
        
        # Calculate survival time
        survival_steps = carnivore.energy / carnivore.energy_decay
        reproduction_cycles = survival_steps / 50  # Assumes reproduction every ~50 steps
        
        print(f"\n🕒 Survival Analysis:")
        print(f"   • Steps until death (no food): {survival_steps:.1f}")
        print(f"   • Potential reproduction cycles: {reproduction_cycles:.1f}")
        print(f"   • Can reproduce immediately: {carnivore.can_reproduce()}")
        
        # Test what happens with different energy decay rates
        print(f"\n⚖️ Energy Decay Rate Analysis:")
        for decay_rate in [0.8, 1.5, 2.5, 3.5]:
            survival = 150 / decay_rate
            repro_cycles = survival / 50
            print(f"   • Decay {decay_rate}: {survival:.1f} steps, {repro_cycles:.1f} reproductions")

def simulate_carnivore_only_scenario():
    """Simulate what happens when only carnivores remain"""
    print(f"\n🦺 Simulating Carnivore-Only Scenario (No Herbivores)")
    print("=" * 60)
    
    env = Phase2NeuralEnvironment(100, 100)
    
    # Remove all herbivores to simulate post-extinction scenario
    original_carnivores = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE])
    env.agents = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE]
    
    print(f"🏁 Starting with {len(env.agents)} carnivores (herbivores removed)")
    
    # Run simulation for 200 steps
    for step in range(200):
        env.step()
        carnivore_count = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE])
        
        if step % 25 == 0:
            avg_energy = sum(a.energy for a in env.agents) / len(env.agents) if env.agents else 0
            print(f"   Step {step:3d}: {carnivore_count:3d} carnivores, avg energy: {avg_energy:.1f}")
    
    final_count = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE])
    print(f"\n📊 Result: Started with {original_carnivores}, ended with {final_count} carnivores")
    print(f"   Population change: {((final_count - original_carnivores) / original_carnivores * 100):+.1f}%")

def test_proposed_fixes():
    """Test different parameter fixes"""
    print(f"\n🛠️ Testing Proposed Fixes")
    print("=" * 40)
    
    fixes = [
        ("Current (Broken)", 0.8, 150, 150),
        ("Fix 1: Higher Decay", 2.5, 150, 150),
        ("Fix 2: Higher Repro Threshold", 0.8, 200, 150),
        ("Fix 3: No Starvation Repro", 0.8, 150, 150),  # Will implement logic change
        ("Fix 4: Combination", 2.0, 180, 150)
    ]
    
    for fix_name, decay, repro_threshold, starting_energy in fixes:
        survival_steps = starting_energy / decay
        can_reproduce_immediately = starting_energy >= repro_threshold
        reproduction_cycles = survival_steps / 50 if survival_steps > 50 else 0
        
        print(f"\n{fix_name}:")
        print(f"   Decay: {decay}, Repro Threshold: {repro_threshold}, Starting: {starting_energy}")
        print(f"   Survival: {survival_steps:.1f} steps")
        print(f"   Immediate reproduction: {'Yes' if can_reproduce_immediately else 'No'}")
        print(f"   Reproduction cycles: {reproduction_cycles:.1f}")

if __name__ == "__main__":
    analyze_carnivore_survival()
    simulate_carnivore_only_scenario()
    test_proposed_fixes()
