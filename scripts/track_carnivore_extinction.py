#!/usr/bin/env python3
"""
Track the critical first 100 steps to see why carnivores die
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import Phase2NeuralEnvironment
from src.core.ecosystem import SpeciesType

def track_early_extinction():
    """Track why carnivores go extinct early"""
    print("⚠️ Tracking Early Carnivore Extinction")
    print("=" * 50)
    
    env = Phase2NeuralEnvironment(100, 100)
    
    print(f"🏁 Starting: 20 herbivores, 8 carnivores")
    
    carnivore_deaths = []
    
    for step in range(100):
        # Track individual carnivores
        alive_carnivores = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive]
        alive_herbivores = [a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive]
        
        if step % 10 == 0:
            if alive_carnivores:
                avg_energy = sum(c.energy for c in alive_carnivores) / len(alive_carnivores)
                avg_starved = sum(getattr(c, 'steps_since_fed', 0) for c in alive_carnivores) / len(alive_carnivores)
                min_energy = min(c.energy for c in alive_carnivores)
                max_energy = max(c.energy for c in alive_carnivores)
                
                print(f"   Step {step:2d}: {len(alive_carnivores)} carnivores, {len(alive_herbivores)} herbivores")
                print(f"            Energy: avg={avg_energy:.1f}, min={min_energy:.1f}, max={max_energy:.1f}")
                print(f"            Avg starved steps: {avg_starved:.1f}")
                
                # Check reproduction eligibility
                can_reproduce = sum(1 for c in alive_carnivores if c.can_reproduce())
                print(f"            Can reproduce: {can_reproduce}/{len(alive_carnivores)}")
            else:
                print(f"   Step {step:2d}: 0 carnivores, {len(alive_herbivores)} herbivores - EXTINCTION!")
                break
        
        prev_carnivore_count = len(alive_carnivores)
        env.step()
        new_carnivore_count = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive])
        
        if new_carnivore_count < prev_carnivore_count:
            deaths = prev_carnivore_count - new_carnivore_count
            carnivore_deaths.append((step, deaths))
            if deaths > 0:
                print(f"      💀 Step {step}: {deaths} carnivore(s) died")
        elif new_carnivore_count > prev_carnivore_count:
            births = new_carnivore_count - prev_carnivore_count
            print(f"      👶 Step {step}: {births} carnivore(s) born")
    
    print(f"\n📊 Death Analysis:")
    total_deaths = sum(deaths for step, deaths in carnivore_deaths)
    print(f"   Total carnivore deaths: {total_deaths}")
    print(f"   Deaths by step: {carnivore_deaths}")

if __name__ == "__main__":
    track_early_extinction()
