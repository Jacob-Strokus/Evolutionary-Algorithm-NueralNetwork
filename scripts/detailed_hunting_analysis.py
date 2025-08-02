#!/usr/bin/env python3
"""
Detailed hunting analysis to understand why carnivores aren't hunting effectively
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import Phase2NeuralEnvironment
from src.core.ecosystem import SpeciesType

def detailed_hunting_analysis():
    """Analyze carnivore-prey interactions in detail"""
    print("🔍 Detailed Hunting Analysis")
    print("=" * 50)
    
    env = Phase2NeuralEnvironment(50, 50)  # Smaller area for more interactions
    
    carnivores = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive]
    herbivores = [a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive]
    
    print(f"🏁 Starting: {len(carnivores)} carnivores, {len(herbivores)} herbivores")
    print(f"📏 Environment: 50x50 (smaller for more interactions)")
    
    if carnivores:
        print(f"🐺 Carnivore stats: Vision={carnivores[0].vision_range}, Speed={carnivores[0].speed}")
    if herbivores:
        print(f"🦌 Herbivore stats: Vision={herbivores[0].vision_range}, Speed={herbivores[0].speed}")
    
    proximity_counts = {"0-1": 0, "1-2": 0, "2-3": 0, "3-5": 0, "5+": 0}
    hunt_attempts = 0
    hunt_successes = 0
    
    for step in range(50):
        # Track distances before step
        for carnivore in env.agents:
            if carnivore.species_type == SpeciesType.CARNIVORE and carnivore.is_alive:
                min_distance = float('inf')
                for prey in env.agents:
                    if prey.species_type == SpeciesType.HERBIVORE and prey.is_alive:
                        distance = carnivore.position.distance_to(prey.position)
                        min_distance = min(min_distance, distance)
                
                if min_distance != float('inf'):
                    if min_distance <= 1.0:
                        proximity_counts["0-1"] += 1
                    elif min_distance <= 2.0:
                        proximity_counts["1-2"] += 1
                    elif min_distance <= 3.0:
                        proximity_counts["2-3"] += 1
                    elif min_distance <= 5.0:
                        proximity_counts["3-5"] += 1
                    else:
                        proximity_counts["5+"] += 1
        
        prev_herbivore_count = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        
        env.step()
        
        new_herbivore_count = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        carnivore_count = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive])
        
        kills_this_step = prev_herbivore_count - new_herbivore_count
        
        if kills_this_step > 0:
            hunt_successes += kills_this_step
            print(f"   Step {step:2d}: 🎯 {kills_this_step} kills! ({carnivore_count} carnivores, {new_herbivore_count} herbivores)")
        
        if step % 10 == 0:
            avg_carnivore_energy = sum(a.energy for a in env.agents if a.species_type == SpeciesType.CARNIVORE and a.is_alive) / max(1, carnivore_count)
            print(f"   Step {step:2d}: {carnivore_count} carnivores, {new_herbivore_count} herbivores, avg energy: {avg_carnivore_energy:.1f}")
        
        if carnivore_count == 0:
            print(f"   All carnivores died at step {step}")
            break
    
    print(f"\n📊 Proximity Analysis (carnivore-prey distances):")
    total_observations = sum(proximity_counts.values())
    for range_name, count in proximity_counts.items():
        percentage = (count / max(1, total_observations)) * 100
        print(f"   {range_name} units: {count:3d} observations ({percentage:5.1f}%)")
    
    print(f"\n🎯 Hunting Results:")
    print(f"   Total kills: {hunt_successes}")
    print(f"   Kills per step: {hunt_successes/50:.2f}")
    print(f"   Close interactions (0-5 units): {sum(list(proximity_counts.values())[:-1])} / {total_observations}")

if __name__ == "__main__":
    detailed_hunting_analysis()
