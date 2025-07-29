#!/usr/bin/env python3
"""
Carnivore Energy Diagnostic Test
Investigate if carnivores are gaining energy without hunting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType
import time

def test_carnivore_energy_mechanics():
    """Test if carnivores gain energy without hunting"""
    print("🔍 Diagnosing Carnivore Energy Mechanics")
    print("=" * 60)
    
    # Create environment
    env = NeuralEnvironment(width=200, height=200, use_neural_agents=True)
    
    # Find carnivores and herbivores
    carnivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.CARNIVORE]
    herbivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE]
    
    if not carnivores:
        print("❌ No carnivores found in environment!")
        return
    
    print(f"🦌 Found {len(herbivores)} herbivores")
    print(f"🐺 Found {len(carnivores)} carnivores")
    print()
    
    # Track a specific carnivore
    test_carnivore = carnivores[0]
    
    print(f"📊 Tracking Carnivore ID: {test_carnivore.id}")
    print(f"   Initial Energy: {test_carnivore.energy:.1f}")
    print(f"   Energy Decay Rate: {test_carnivore.energy_decay}")
    print(f"   Max Energy: {test_carnivore.max_energy}")
    print(f"   Successful Hunts: {test_carnivore.lifetime_successful_hunts}")
    print()
    
    # Separate carnivores from all herbivores
    print("🚫 Moving carnivore far from all herbivores...")
    test_carnivore.position.x = 10
    test_carnivore.position.y = 10
    
    # Move all herbivores to opposite corner
    for herb in herbivores:
        herb.position.x = 190
        herb.position.y = 190
    
    print("   Carnivore position: (10, 10)")
    print("   All herbivores moved to: (190, 190)")
    print()
    
    # Track energy changes over time
    energy_history = []
    hunt_history = []
    
    print("🔄 Running isolated simulation (carnivore cannot hunt)...")
    print("Step | Energy | Change | Hunts | Notes")
    print("-" * 50)
    
    for step in range(20):
        # Record state before step
        energy_before = test_carnivore.energy
        hunts_before = test_carnivore.lifetime_successful_hunts
        
        # Run one step
        env.step()
        
        # Record state after step
        energy_after = test_carnivore.energy
        hunts_after = test_carnivore.lifetime_successful_hunts
        
        energy_change = energy_after - energy_before
        hunt_change = hunts_after - hunts_before
        
        energy_history.append(energy_after)
        hunt_history.append(hunts_after)
        
        # Notes
        notes = ""
        if hunt_change > 0:
            notes += "🎯 HUNT! "
        if energy_change > 0:
            notes += "⚡ ENERGY GAIN! "
        if abs(energy_change + test_carnivore.energy_decay) < 0.001:  # Account for floating point precision
            notes += "✅ Normal decay"
        elif energy_change > -test_carnivore.energy_decay:
            notes += f"❓ Less decay than expected"
        elif energy_change < -test_carnivore.energy_decay:
            notes += f"❓ More decay than expected"
        
        # Check distance to nearest prey
        nearest_prey = env.find_nearest_prey(test_carnivore)
        if nearest_prey:
            distance = test_carnivore.position.distance_to(nearest_prey.position)
            if distance < 10:
                notes += f" (prey at {distance:.1f})"
        
        print(f"{step:4d} | {energy_after:6.1f} | {energy_change:+6.1f} | {hunts_after:5d} | {notes}")
        
        # Stop if carnivore dies
        if not test_carnivore.is_alive:
            print("💀 Carnivore died!")
            break
    
    print()
    print("📈 Analysis:")
    
    # Check if energy increased without hunting
    total_hunts = test_carnivore.lifetime_successful_hunts - hunt_history[0] if hunt_history else 0
    initial_energy = energy_history[0] if energy_history else test_carnivore.energy
    final_energy = test_carnivore.energy
    
    print(f"   Initial Energy: {initial_energy:.1f}")
    print(f"   Final Energy: {final_energy:.1f}")
    print(f"   Total Hunts: {total_hunts}")
    print(f"   Expected Energy Loss: {len(energy_history) * test_carnivore.energy_decay:.1f}")
    
    if total_hunts == 0 and final_energy > initial_energy:
        print("❌ BUG DETECTED: Carnivore gained energy without hunting!")
        return False
    elif total_hunts == 0 and abs(final_energy - (initial_energy - (len(energy_history) * test_carnivore.energy_decay))) < 0.1:
        print("✅ Normal behavior: Only energy decay, no hunting")
        return True
    elif total_hunts > 0:
        print(f"✅ Normal behavior: {total_hunts} successful hunts")
        return True
    else:
        print("❓ Unexpected energy pattern - needs investigation")
        print(f"   Expected final energy: {initial_energy - (len(energy_history) * test_carnivore.energy_decay):.1f}")
        print(f"   Actual final energy: {final_energy:.1f}")
        return False

def test_carnivore_food_access():
    """Test if carnivores can access food sources like herbivores"""
    print("\n🌾 Testing Carnivore Food Access")
    print("=" * 40)
    
    env = NeuralEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Find a carnivore
    carnivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.CARNIVORE]
    if not carnivores:
        print("❌ No carnivores found!")
        return
    
    test_carnivore = carnivores[0]
    
    # Place carnivore next to food
    if env.food_sources:
        food = env.food_sources[0]
        test_carnivore.position.x = food.position.x + 1
        test_carnivore.position.y = food.position.y + 1
        
        print(f"🐺 Carnivore at: ({test_carnivore.position.x:.1f}, {test_carnivore.position.y:.1f})")
        print(f"🌾 Food at: ({food.position.x:.1f}, {food.position.y:.1f})")
        print(f"📏 Distance: {test_carnivore.position.distance_to(food.position):.1f}")
        print(f"⚡ Carnivore energy before: {test_carnivore.energy:.1f}")
        print(f"🌾 Food available: {food.is_available}")
        
        # Run a few steps
        for i in range(5):
            energy_before = test_carnivore.energy
            food_available_before = food.is_available
            
            env.step()
            
            energy_after = test_carnivore.energy
            food_available_after = food.is_available
            
            energy_change = energy_after - energy_before
            food_consumed = food_available_before and not food_available_after
            
            print(f"Step {i+1}: Energy {energy_change:+.1f}, Food consumed: {food_consumed}")
            
            if food_consumed and energy_change > 0:
                print("❌ BUG: Carnivore consumed plant food!")
                return False
    
    print("✅ Carnivores cannot consume plant food (correct)")
    return True

if __name__ == "__main__":
    try:
        print("🧬 Carnivore Energy Diagnostic Suite")
        print("=" * 70)
        
        result1 = test_carnivore_energy_mechanics()
        result2 = test_carnivore_food_access()
        
        print("\n🏁 Final Results:")
        print(f"   Energy Mechanics Test: {'✅ PASS' if result1 else '❌ FAIL'}")
        print(f"   Food Access Test: {'✅ PASS' if result2 else '❌ FAIL'}")
        
        if result1 and result2:
            print("\n🎉 All tests passed! Carnivore energy mechanics are working correctly.")
        else:
            print("\n⚠️ Issues detected in carnivore energy system!")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
