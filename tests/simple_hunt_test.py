#!/usr/bin/env python3
"""
Simple hunting test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType

def test_simple_hunt():
    print("🎯 Simple Hunt Test")
    
    env = NeuralEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Find carnivore and herbivore
    carnivore = None
    herbivore = None
    
    for agent in env.agents:
        if isinstance(agent, NeuralAgent):
            if agent.species_type == SpeciesType.CARNIVORE and not carnivore:
                carnivore = agent
            elif agent.species_type == SpeciesType.HERBIVORE and not herbivore:
                herbivore = agent
        if carnivore and herbivore:
            break
    
    if not carnivore or not herbivore:
        print("❌ Need both carnivore and herbivore!")
        return
    
    print(f"🐺 Carnivore {carnivore.id}: Energy {carnivore.energy:.1f}")
    print(f"🦌 Herbivore {herbivore.id}: Energy {herbivore.energy:.1f}, Alive: {herbivore.is_alive}")
    
    # Place close together
    carnivore.position.x = 25
    carnivore.position.y = 25
    herbivore.position.x = 26
    herbivore.position.y = 25
    
    print(f"📏 Distance: {carnivore.position.distance_to(herbivore.position):.2f}")
    
    # Test hunt directly
    print("\n🔬 Testing hunt method directly:")
    energy_before = carnivore.energy
    alive_before = herbivore.is_alive
    
    env._handle_neural_feeding(carnivore)
    
    energy_after = carnivore.energy
    alive_after = herbivore.is_alive
    
    print(f"Energy: {energy_before:.1f} → {energy_after:.1f} (change: {energy_after - energy_before:+.1f})")
    print(f"Prey alive: {alive_before} → {alive_after}")
    
    if energy_after > energy_before and alive_after:
        print("❌ BUG: Energy gained but prey still alive!")
    elif energy_after > energy_before and not alive_after:
        print("✅ Normal hunt: Energy gained and prey killed")
    else:
        print("ℹ️ No hunt occurred")
    
    print("\n🔄 Testing full step:")
    
    # Reset for full step test
    carnivore.energy = 150
    herbivore.energy = 150
    herbivore.is_alive = True
    
    energy_before = carnivore.energy
    alive_before = herbivore.is_alive
    herbivore_count_before = len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE and a.is_alive])
    
    env.step()
    
    energy_after = carnivore.energy
    alive_after = herbivore.is_alive
    herbivore_count_after = len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE and a.is_alive])
    
    print(f"Energy: {energy_before:.1f} → {energy_after:.1f} (change: {energy_after - energy_before:+.1f})")
    print(f"Our herbivore alive: {alive_before} → {alive_after}")
    print(f"Total herbivores: {herbivore_count_before} → {herbivore_count_after}")
    
    if energy_after > energy_before and herbivore_count_after == herbivore_count_before:
        print("❌ BUG: Energy gained but no herbivores died!")
    elif energy_after > energy_before and herbivore_count_after < herbivore_count_before:
        print("✅ Normal: Energy gained and herbivore(s) died")
    else:
        print("ℹ️ No energy gain")

if __name__ == "__main__":
    try:
        test_simple_hunt()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
