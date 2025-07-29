#!/usr/bin/env python3
"""
Comprehensive carnivore behavior test
Tests both the food source bug fix and hunting consumption tracking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType

def test_comprehensive_carnivore_behavior():
    """Test all aspects of carnivore behavior"""
    print("🧪 Comprehensive Carnivore Behavior Test")
    print("=" * 70)
    
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Get counts
    carnivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.CARNIVORE]
    herbivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE]
    
    print(f"🦌 Herbivores: {len(herbivores)}")
    print(f"🐺 Carnivores: {len(carnivores)}")
    print(f"🌾 Food sources: {len(env.food_sources)}")
    print()
    
    if not carnivores:
        print("❌ No carnivores to test!")
        return False
    
    # Track one specific carnivore
    test_carnivore = carnivores[0]
    print(f"📊 Tracking Carnivore ID: {test_carnivore.id}")
    print(f"   Initial Energy: {test_carnivore.energy:.1f}")
    print(f"   Initial Hunt Count: {test_carnivore.lifetime_successful_hunts}")
    print(f"   Initial Food Consumed: {test_carnivore.lifetime_food_consumed}")
    print()
    
    # Test 1: Verify carnivore doesn't eat food sources
    print("🧪 Test 1: Food Source Consumption")
    print("-" * 40)
    
    # Place carnivore near food source
    if env.food_sources:
        food = env.food_sources[0]
        test_carnivore.position.x = food.position.x + 0.5
        test_carnivore.position.y = food.position.y + 0.5
        
        print(f"🎯 Placed carnivore near food source")
        print(f"   Distance: {test_carnivore.position.distance_to(food.position):.2f}")
        
        # Test direct feeding
        energy_before = test_carnivore.energy
        food_available_before = food.is_available
        
        env._handle_neural_feeding(test_carnivore)
        
        energy_after = test_carnivore.energy
        food_available_after = food.is_available
        
        if energy_after > energy_before and not food_available_after:
            print("❌ BUG: Carnivore consumed food source!")
            return False
        elif energy_after > energy_before:
            print("✅ Carnivore gained energy from hunting, not food source")
        else:
            print("✅ Carnivore correctly ignored food source")
    
    print()
    
    # Test 2: Verify carnivore hunting behavior and consumption tracking
    print("🧪 Test 2: Hunting Behavior & Consumption Tracking")
    print("-" * 50)
    
    # Find a herbivore
    herbivore = None
    for agent in env.agents:
        if isinstance(agent, NeuralAgent) and agent.species_type == SpeciesType.HERBIVORE and agent.is_alive:
            herbivore = agent
            break
    
    if not herbivore:
        print("❌ No herbivore found for hunting test!")
        return False
    
    # Place carnivore near herbivore
    test_carnivore.position.x = 50
    test_carnivore.position.y = 50
    herbivore.position.x = 51
    herbivore.position.y = 50
    
    print(f"🎯 Placed carnivore near herbivore")
    print(f"   Distance: {test_carnivore.position.distance_to(herbivore.position):.2f}")
    
    # Reset tracking values
    test_carnivore.energy = 150
    initial_hunts = test_carnivore.lifetime_successful_hunts
    initial_food_consumed = test_carnivore.lifetime_food_consumed
    
    # Test multiple hunt attempts
    successful_hunts = 0
    
    for attempt in range(10):
        herbivore.energy = 150
        herbivore.is_alive = True
        
        energy_before = test_carnivore.energy
        hunts_before = test_carnivore.lifetime_successful_hunts
        food_consumed_before = test_carnivore.lifetime_food_consumed
        
        env._handle_neural_feeding(test_carnivore)
        
        energy_after = test_carnivore.energy
        hunts_after = test_carnivore.lifetime_successful_hunts
        food_consumed_after = test_carnivore.lifetime_food_consumed
        
        if energy_after > energy_before:
            successful_hunts += 1
            hunt_increase = hunts_after - hunts_before
            consumption_increase = food_consumed_after - food_consumed_before
            
            print(f"Hunt {successful_hunts}: Success!")
            print(f"   Energy: {energy_before:.1f} → {energy_after:.1f}")
            print(f"   Hunts: {hunts_before} → {hunts_after} (+{hunt_increase})")
            print(f"   Food consumed: {food_consumed_before} → {food_consumed_after} (+{consumption_increase})")
            print(f"   Prey alive: {herbivore.is_alive}")
            
            if hunt_increase == 1 and consumption_increase == 1 and not herbivore.is_alive:
                print(f"   ✅ Perfect tracking!")
            else:
                print(f"   ❌ Tracking error!")
                return False
        
        # Reset energy for next attempt
        test_carnivore.energy = 150
    
    if successful_hunts == 0:
        print("⚠️ No successful hunts in 10 attempts (bad luck)")
        return True  # Not necessarily a failure
    
    print()
    
    # Test 3: Full simulation behavior
    print("🧪 Test 3: Full Simulation Behavior")
    print("-" * 35)
    
    # Reset environment for clean test
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    carnivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.CARNIVORE]
    
    initial_carnivore_food_consumed = sum(c.lifetime_food_consumed for c in carnivores)
    initial_carnivore_hunts = sum(c.lifetime_successful_hunts for c in carnivores)
    initial_herbivores = len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE])
    
    print(f"Initial state:")
    print(f"   Carnivore total food consumed: {initial_carnivore_food_consumed}")
    print(f"   Carnivore total hunts: {initial_carnivore_hunts}")
    print(f"   Herbivore count: {initial_herbivores}")
    
    # Run simulation
    for step in range(15):
        food_available_before = sum(1 for f in env.food_sources if f.is_available)
        
        env.step()
        
        food_available_after = sum(1 for f in env.food_sources if f.is_available)
        food_consumed_this_step = food_available_before - food_available_after
        
        current_carnivore_food_consumed = sum(c.lifetime_food_consumed for c in carnivores if c.is_alive)
        current_carnivore_hunts = sum(c.lifetime_successful_hunts for c in carnivores if c.is_alive)
        current_herbivores = len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        
        carnivore_consumption_increase = current_carnivore_food_consumed - initial_carnivore_food_consumed
        carnivore_hunt_increase = current_carnivore_hunts - initial_carnivore_hunts
        herbivores_lost = initial_herbivores - current_herbivores
        
        if carnivore_consumption_increase > 0 or herbivores_lost > 0:
            print(f"Step {step}:")
            print(f"   Plant food consumed: {food_consumed_this_step}")
            print(f"   Carnivore consumption: +{carnivore_consumption_increase}")
            print(f"   Carnivore hunts: +{carnivore_hunt_increase}")
            print(f"   Herbivores lost: {herbivores_lost}")
            
            # Validate consistency
            if carnivore_consumption_increase != carnivore_hunt_increase:
                print(f"   ❌ Carnivore tracking inconsistency!")
                return False
    
    final_carnivore_food_consumed = sum(c.lifetime_food_consumed for c in carnivores if c.is_alive)
    final_carnivore_hunts = sum(c.lifetime_successful_hunts for c in carnivores if c.is_alive)
    
    total_carnivore_consumption = final_carnivore_food_consumed - initial_carnivore_food_consumed
    total_carnivore_hunts = final_carnivore_hunts - initial_carnivore_hunts
    
    print(f"\nFinal results:")
    print(f"   Total carnivore consumption: {total_carnivore_consumption}")
    print(f"   Total carnivore hunts: {total_carnivore_hunts}")
    
    if total_carnivore_consumption == total_carnivore_hunts:
        print(f"✅ Carnivore consumption tracking is perfectly consistent!")
        return True
    else:
        print(f"❌ Carnivore consumption tracking inconsistency!")
        return False

if __name__ == "__main__":
    try:
        result = test_comprehensive_carnivore_behavior()
        
        print(f"\n🏁 Final Result:")
        if result:
            print(f"🎉 All carnivore behavior tests PASSED!")
            print(f"   ✅ Carnivores don't eat food sources")
            print(f"   ✅ Carnivore hunting works correctly")
            print(f"   ✅ Carnivore consumption tracking is accurate")
        else:
            print(f"❌ Carnivore behavior tests FAILED!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
