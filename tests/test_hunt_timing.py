#!/usr/bin/env python3
"""
Direct hunt tracking to catch the exact bug
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType
import random

def test_exact_hunt_timing():
    """Test exact timing of hunts vs prey deaths"""
    print("🎯 Testing Exact Hunt Timing")
    print("=" * 50)
    
    # Seed for reproducible results
    random.seed(42)
    
    env = NeuralEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Find one carnivore
    carnivore = None
    for agent in env.agents:
        if isinstance(agent, NeuralAgent) and agent.species_type == SpeciesType.CARNIVORE:
            carnivore = agent
            break
    
    if not carnivore:
        print("❌ No carnivore found!")
        return
    
    print(f"🐺 Testing Carnivore ID: {carnivore.id}")
    print(f"   Starting energy: {carnivore.energy:.1f}")
    print(f"   Starting hunts: {carnivore.lifetime_successful_hunts}")
    print()
    
    # Track herbivore deaths and carnivore energy gains
    hunt_log = []
    death_log = []
    
    for step in range(30):
        # Before step
        energy_before = carnivore.energy
        hunts_before = carnivore.lifetime_successful_hunts
        herbivores_before = [(a.id, a.is_alive, a.energy) for a in env.agents 
                           if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE]
        
        # Run step
        env.step()
        
        # After step
        energy_after = carnivore.energy
        hunts_after = carnivore.lifetime_successful_hunts
        herbivores_after = [(a.id, a.is_alive, a.energy) for a in env.agents 
                          if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE]
        
        # Check for energy gain
        energy_gain = energy_after - energy_before
        hunt_gain = hunts_after - hunts_before
        
        if energy_gain > 0:
            hunt_log.append({
                'step': step,
                'energy_gain': energy_gain,
                'hunt_count_gain': hunt_gain,
                'energy_before': energy_before,
                'energy_after': energy_after
            })
            print(f"Step {step}: Carnivore gained {energy_gain:.1f} energy (hunt count +{hunt_gain})")
        
        # Check for herbivore deaths
        for (id_before, alive_before, energy_before), (id_after, alive_after, energy_after) in zip(herbivores_before, herbivores_after):
            if id_before == id_after and alive_before and not alive_after:
                death_log.append({
                    'step': step,
                    'herbivore_id': id_before,
                    'energy_when_died': energy_before
                })
                print(f"Step {step}: Herbivore {id_before} died (had {energy_before:.1f} energy)")
        
        if not carnivore.is_alive:
            print(f"Step {step}: Carnivore died")
            break
    
    print(f"\n📊 Analysis:")
    print(f"   Total energy gained: {sum(h['energy_gain'] for h in hunt_log):.1f}")
    print(f"   Total hunts: {sum(h['hunt_count_gain'] for h in hunt_log)}")
    print(f"   Total herbivore deaths: {len(death_log)}")
    
    print(f"\n🔍 Detailed Logs:")
    print("Hunt Log:")
    for hunt in hunt_log:
        print(f"   Step {hunt['step']}: +{hunt['energy_gain']:.1f} energy, +{hunt['hunt_count_gain']} hunts")
    
    print("Death Log:")
    for death in death_log:
        print(f"   Step {death['step']}: Herbivore {death['herbivore_id']} died ({death['energy_when_died']:.1f} energy)")
    
    # Check for mismatches
    hunt_steps = set(h['step'] for h in hunt_log)
    death_steps = set(d['step'] for d in death_log)
    
    print(f"\n🚨 Mismatch Analysis:")
    hunts_without_deaths = hunt_steps - death_steps
    deaths_without_hunts = death_steps - hunt_steps
    
    if hunts_without_deaths:
        print(f"❌ Steps with energy gains but no deaths: {hunts_without_deaths}")
        for step in hunts_without_deaths:
            hunt_info = next(h for h in hunt_log if h['step'] == step)
            print(f"   Step {step}: Gained {hunt_info['energy_gain']:.1f} energy without killing")
    
    if deaths_without_hunts:
        print(f"⚠️ Steps with deaths but no energy gains: {deaths_without_hunts}")
    
    if not hunts_without_deaths and not deaths_without_hunts:
        print("✅ All energy gains correspond to herbivore deaths")
    
    return len(hunts_without_deaths) == 0

def test_hunt_method_directly():
    """Test the hunt method in isolation"""
    print("\n🧪 Testing Hunt Method Directly")
    print("=" * 40)
    
    env = NeuralEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Get a carnivore and herbivore
    carnivore = None
    herbivore = None
    
    for agent in env.agents:
        if isinstance(agent, NeuralAgent):
            if agent.species_type == SpeciesType.CARNIVORE and not carnivore:
                carnivore = agent
            elif agent.species_type == SpeciesType.HERBIVORE and not herbivore:
                herbivore = agent
    
    if not carnivore or not herbivore:
        print("❌ Need both carnivore and herbivore!")
        return False
    
    # Place them close together
    carnivore.position.x = 25
    carnivore.position.y = 25
    herbivore.position.x = 26
    herbivore.position.y = 25
    
    print(f"🐺 Carnivore: {carnivore.id}, Energy: {carnivore.energy:.1f}")
    print(f"🦌 Herbivore: {herbivore.id}, Energy: {herbivore.energy:.1f}")
    
    # Force multiple hunts to see if there's a pattern
    results = []
    
    for attempt in range(10):
        # Reset states
        carnivore.energy = 150
        herbivore.energy = 150
        herbivore.is_alive = True
        
        energy_before = carnivore.energy
        prey_alive_before = herbivore.is_alive
        
        # Call feeding method
        env._handle_neural_feeding(carnivore)
        
        energy_after = carnivore.energy
        prey_alive_after = herbivore.is_alive
        
        energy_gain = energy_after - energy_before
        prey_killed = prey_alive_before and not prey_alive_after
        
        results.append({
            'attempt': attempt,
            'energy_gain': energy_gain,
            'prey_killed': prey_killed,
            'bug': energy_gain > 0 and not prey_killed
        })
        
        if energy_gain > 0:
            print(f"Attempt {attempt}: Energy +{energy_gain:.1f}, Prey killed: {prey_killed}")
    
    bugs = [r for r in results if r['bug']]
    successful_hunts = [r for r in results if r['energy_gain'] > 0 and r['prey_killed']]
    
    print(f"\nResults:")
    print(f"   Bugs (energy without kill): {len(bugs)}")
    print(f"   Successful hunts: {len(successful_hunts)}")
    print(f"   Bug rate: {len(bugs) / max(1, len(bugs) + len(successful_hunts)) * 100:.1f}%")
    
    return len(bugs) == 0

if __name__ == "__main__":
    try:
        result1 = test_exact_hunt_timing()
        result2 = test_hunt_method_directly()
        
        print(f"\n🏁 Final Results:")
        print(f"   Timing test: {'✅ PASS' if result1 else '❌ FAIL'}")
        print(f"   Direct test: {'✅ PASS' if result2 else '❌ FAIL'}")
        
        if not result1 or not result2:
            print("\n❌ Energy gain bug confirmed!")
        else:
            print("\n✅ No bugs detected!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
