#!/usr/bin/env python3
"""
Test carnivore hunting effectiveness
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import Phase2NeuralEnvironment
from src.core.ecosystem import SpeciesType

def test_hunting_effectiveness():
    """Test how well carnivores can hunt"""
    print("🏹 Testing Carnivore Hunting Effectiveness")
    print("=" * 50)
    
    env = Phase2NeuralEnvironment(100, 100)
    
    carnivores = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE]
    herbivores = [a for a in env.agents if a.species_type == SpeciesType.HERBIVORE]
    
    print(f"🏁 Starting: {len(carnivores)} carnivores, {len(herbivores)} herbivores")
    
    hunts_attempted = 0
    hunts_successful = 0
    
    for step in range(100):
        prev_herbivore_count = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        
        env.step()
        
        new_herbivore_count = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        carnivore_count = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive])
        
        kills_this_step = prev_herbivore_count - new_herbivore_count
        
        if step % 10 == 0:
            avg_carnivore_energy = sum(a.energy for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive) / max(1, carnivore_count)
            avg_starved = sum(getattr(a, 'steps_since_fed', 0) for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive) / max(1, carnivore_count)
            
            print(f"   Step {step:2d}: {carnivore_count} carnivores, {new_herbivore_count} herbivores")
            print(f"             Carnivore avg energy: {avg_carnivore_energy:.1f}, avg starved: {avg_starved:.1f}")
            
            if kills_this_step > 0:
                print(f"             Kills this step: {kills_this_step}")
                hunts_successful += kills_this_step
        
        if carnivore_count == 0:
            print(f"   All carnivores died at step {step}")
            break
    
    print(f"\n📊 Hunting Results:")
    print(f"   Hunt success rate: {hunts_successful}/{100} steps = {hunts_successful/100*100:.1f}% steps with kills")
    print(f"   Average kills per step: {hunts_successful/100:.2f}")

if __name__ == "__main__":
    test_hunting_effectiveness()
