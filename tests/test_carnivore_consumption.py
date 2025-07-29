#!/usr/bin/env python3
"""
Test carnivore food consumption tracking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType

def test_carnivore_consumption_tracking():
    """Test that carnivores properly track their consumption of herbivores"""
    print("🍽️ Testing Carnivore Consumption Tracking")
    print("=" * 60)
    
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
        return False
    
    print(f"🐺 Carnivore {carnivore.id}")
    print(f"   Initial Energy: {carnivore.energy:.1f}")
    print(f"   Initial Food Consumed: {carnivore.lifetime_food_consumed}")
    print(f"   Initial Hunt Count: {carnivore.lifetime_successful_hunts}")
    
    print(f"🦌 Herbivore {herbivore.id}")
    print(f"   Initial Energy: {herbivore.energy:.1f}")
    print(f"   Initial Alive: {herbivore.is_alive}")
    
    # Place them close together to guarantee a hunt
    carnivore.position.x = 25
    carnivore.position.y = 25
    herbivore.position.x = 26
    herbivore.position.y = 25
    
    distance = carnivore.position.distance_to(herbivore.position)
    print(f"📏 Distance: {distance:.2f}")
    
    # Test multiple hunts to ensure consistent tracking
    hunt_attempts = 0
    successful_hunts = 0
    
    for attempt in range(10):
        # Reset states for each attempt
        carnivore.energy = 150
        herbivore.energy = 150
        herbivore.is_alive = True
        
        energy_before = carnivore.energy
        food_consumed_before = carnivore.lifetime_food_consumed
        hunts_before = carnivore.lifetime_successful_hunts
        
        # Attempt hunt
        env._handle_neural_feeding(carnivore)
        hunt_attempts += 1
        
        energy_after = carnivore.energy
        food_consumed_after = carnivore.lifetime_food_consumed
        hunts_after = carnivore.lifetime_successful_hunts
        
        energy_gained = energy_after - energy_before
        food_count_increase = food_consumed_after - food_consumed_before
        hunt_count_increase = hunts_after - hunts_before
        
        if energy_gained > 0:  # Successful hunt
            successful_hunts += 1
            print(f"Attempt {attempt}: ✅ Hunt successful")
            print(f"   Energy: {energy_before:.1f} → {energy_after:.1f} (+{energy_gained:.1f})")
            print(f"   Food consumed: {food_consumed_before} → {food_consumed_after} (+{food_count_increase})")
            print(f"   Hunt count: {hunts_before} → {hunts_after} (+{hunt_count_increase})")
            print(f"   Prey alive: {herbivore.is_alive}")
            
            # Validate tracking
            if food_count_increase == 1 and hunt_count_increase == 1:
                print(f"   ✅ Tracking correct: Both food consumed and hunt count increased")
            elif food_count_increase == 0:
                print(f"   ❌ Food consumption not tracked!")
                return False
            elif hunt_count_increase == 0:
                print(f"   ❌ Hunt count not tracked!")
                return False
            else:
                print(f"   ❓ Unexpected tracking values")
                return False
        else:
            print(f"Attempt {attempt}: 🎲 Hunt failed (random chance)")
    
    print(f"\n📊 Summary:")
    print(f"   Hunt attempts: {hunt_attempts}")
    print(f"   Successful hunts: {successful_hunts}")
    print(f"   Success rate: {(successful_hunts/hunt_attempts)*100:.1f}%")
    
    if successful_hunts > 0:
        print(f"✅ Carnivore consumption tracking is working correctly!")
        return True
    else:
        print(f"⚠️ No successful hunts occurred to test tracking")
        return False

def test_consumption_in_simulation():
    """Test consumption tracking in a full simulation"""
    print("\n🌍 Testing Consumption in Full Simulation")
    print("=" * 50)
    
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Get all carnivores
    carnivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.CARNIVORE]
    herbivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE]
    
    print(f"🦌 Initial herbivores: {len(herbivores)}")
    print(f"🐺 Initial carnivores: {len(carnivores)}")
    
    # Track initial consumption
    initial_total_food_consumed = sum(c.lifetime_food_consumed for c in carnivores)
    initial_total_hunts = sum(c.lifetime_successful_hunts for c in carnivores)
    
    print(f"📊 Initial total food consumed: {initial_total_food_consumed}")
    print(f"🎯 Initial total hunts: {initial_total_hunts}")
    
    # Run simulation for several steps
    print(f"\n🔄 Running simulation...")
    
    for step in range(10):
        herbivores_before = len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        
        total_food_consumed_before = sum(c.lifetime_food_consumed for c in carnivores if c.is_alive)
        total_hunts_before = sum(c.lifetime_successful_hunts for c in carnivores if c.is_alive)
        
        env.step()
        
        herbivores_after = len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        
        total_food_consumed_after = sum(c.lifetime_food_consumed for c in carnivores if c.is_alive)
        total_hunts_after = sum(c.lifetime_successful_hunts for c in carnivores if c.is_alive)
        
        herbivores_killed = herbivores_before - herbivores_after
        food_consumed_increase = total_food_consumed_after - total_food_consumed_before
        hunts_increase = total_hunts_after - total_hunts_before
        
        if herbivores_killed > 0 or food_consumed_increase > 0:
            print(f"Step {step}: Herbivores killed: {herbivores_killed}, Food consumed: +{food_consumed_increase}, Hunts: +{hunts_increase}")
            
            if food_consumed_increase == hunts_increase and food_consumed_increase == herbivores_killed:
                print(f"   ✅ Perfect tracking: All metrics match")
            elif food_consumed_increase != hunts_increase:
                print(f"   ❌ Mismatch: Food consumed ({food_consumed_increase}) != Hunts ({hunts_increase})")
                return False
            elif food_consumed_increase != herbivores_killed:
                print(f"   ⚠️ Note: Tracking ({food_consumed_increase}) != Deaths ({herbivores_killed}) - may be due to multiple carnivores hunting same prey")
    
    final_total_food_consumed = sum(c.lifetime_food_consumed for c in carnivores if c.is_alive)
    final_total_hunts = sum(c.lifetime_successful_hunts for c in carnivores if c.is_alive)
    
    total_food_increase = final_total_food_consumed - initial_total_food_consumed
    total_hunt_increase = final_total_hunts - initial_total_hunts
    
    print(f"\n📈 Final Results:")
    print(f"   Total food consumed increase: {total_food_increase}")
    print(f"   Total hunt increase: {total_hunt_increase}")
    
    if total_food_increase == total_hunt_increase:
        print(f"✅ Consumption tracking is consistent across simulation!")
        return True
    else:
        print(f"❌ Consumption tracking inconsistency detected!")
        return False

if __name__ == "__main__":
    try:
        result1 = test_carnivore_consumption_tracking()
        result2 = test_consumption_in_simulation()
        
        print(f"\n🏁 Final Results:")
        print(f"   Direct tracking test: {'✅ PASS' if result1 else '❌ FAIL'}")
        print(f"   Simulation tracking test: {'✅ PASS' if result2 else '❌ FAIL'}")
        
        if result1 and result2:
            print(f"\n🎉 Carnivore consumption tracking is working perfectly!")
        else:
            print(f"\n⚠️ Issues detected in consumption tracking!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
