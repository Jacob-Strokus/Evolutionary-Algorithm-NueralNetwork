#!/usr/bin/env python3
"""
Debug the exact hunting mechanism
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType
import random

def debug_hunting_logic():
    """Debug the exact hunting logic step by step"""
    print("🔍 Debugging Hunting Logic")
    print("=" * 50)
    
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
    
    if not carnivore or not herbivore:
        print("❌ Need both carnivore and herbivore!")
        return
        
    # Place them very close
    carnivore.position.x = 25
    carnivore.position.y = 25
    herbivore.position.x = 26  # Very close
    herbivore.position.y = 25
    
    print(f"🐺 Carnivore: ID {carnivore.id}")
    print(f"   Energy: {carnivore.energy:.1f}")
    print(f"   Position: ({carnivore.position.x:.1f}, {carnivore.position.y:.1f})")
    print(f"   Hunts: {carnivore.lifetime_successful_hunts}")
    
    print(f"🦌 Herbivore: ID {herbivore.id}")
    print(f"   Energy: {herbivore.energy:.1f}")
    print(f"   Position: ({herbivore.position.x:.1f}, {herbivore.position.y:.1f})")
    print(f"   Alive: {herbivore.is_alive}")
    
    distance = carnivore.position.distance_to(herbivore.position)
    print(f"📏 Distance: {distance:.2f}")
    print()
    
    # Step through the hunting logic manually
    print("🎯 Manually stepping through hunting logic:")
    
    # Find nearest prey
    prey = env.find_nearest_prey(carnivore)
    print(f"1. Nearest prey found: {prey is not None}")
    if prey:
        print(f"   Prey ID: {prey.id}, Energy: {prey.energy:.1f}")
        
    # Check distance
    if prey and carnivore.position.distance_to(prey.position) < 4.0:
        print(f"2. Within hunting range: YES (distance: {distance:.2f} < 4.0)")
        
        # Calculate hunting success
        hunt_success_chance = min(0.7, carnivore.energy / 120)
        print(f"3. Hunt success chance: {hunt_success_chance:.3f}")
        
        # Force a successful hunt for testing
        print("4. Forcing successful hunt for testing...")
        
        energy_before = carnivore.energy
        prey_alive_before = prey.is_alive
        hunts_before = carnivore.lifetime_successful_hunts
        
        energy_gained = int(prey.energy * 0.8)
        print(f"   Energy to be gained: {energy_gained}")
        
        # Call successful_hunt
        carnivore.successful_hunt(energy_gained)
        
        energy_after = carnivore.energy
        hunts_after = carnivore.lifetime_successful_hunts
        
        print(f"   Energy before: {energy_before:.1f}")
        print(f"   Energy after: {energy_after:.1f}")
        print(f"   Energy change: {energy_after - energy_before:.1f}")
        print(f"   Hunts before: {hunts_before}")
        print(f"   Hunts after: {hunts_after}")
        
        # Kill the prey (this should happen after energy gain)
        prey.is_alive = False
        print(f"   Prey killed: {not prey.is_alive}")
        
    print("\n🔄 Now testing the actual _handle_neural_feeding method:")
    
    # Reset for actual test
    carnivore.energy = 150
    herbivore.energy = 150
    herbivore.is_alive = True
    carnivore.lifetime_successful_hunts = 0
    
    energy_before = carnivore.energy
    prey_alive_before = herbivore.is_alive
    hunts_before = carnivore.lifetime_successful_hunts
    
    # Use the actual feeding method
    env._handle_neural_feeding(carnivore)
    
    energy_after = carnivore.energy
    prey_alive_after = herbivore.is_alive
    hunts_after = carnivore.lifetime_successful_hunts
    
    print(f"Energy: {energy_before:.1f} → {energy_after:.1f} (change: {energy_after - energy_before:+.1f})")
    print(f"Prey alive: {prey_alive_before} → {prey_alive_after}")
    print(f"Hunt count: {hunts_before} → {hunts_after}")
    
    if energy_after > energy_before and prey_alive_after:
        print("❌ BUG CONFIRMED: Energy gained but prey not killed!")
    elif energy_after > energy_before and not prey_alive_after:
        print("✅ Normal hunt: Energy gained and prey killed")
    else:
        print("ℹ️ No hunt occurred (random chance)")

def test_energy_calculation():
    """Test the energy calculation specifically"""
    print("\n🧮 Testing Energy Calculation")
    print("=" * 40)
    
    # Create a test agent
    from src.neural.neural_network import NeuralNetworkConfig
    from src.core.ecosystem import Position
    agent = NeuralAgent(SpeciesType.CARNIVORE, Position(0, 0), 999)
    agent.energy = 150
    
    print(f"Starting energy: {agent.energy:.1f}")
    print(f"Max energy: {agent.max_energy}")
    
    # Test various energy gains
    test_gains = [50, 100, 120, 150, 200]
    
    for gain in test_gains:
        original_energy = agent.energy
        agent.successful_hunt(gain)
        final_energy = agent.energy
        actual_gain = final_energy - original_energy
        
        print(f"Attempted gain: {gain}, Actual gain: {actual_gain:.1f}")
        
        # Reset for next test
        agent.energy = 150

if __name__ == "__main__":
    try:
        debug_hunting_logic()
        test_energy_calculation()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
