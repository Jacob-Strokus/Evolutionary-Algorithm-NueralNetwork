#!/usr/bin/env python3
"""
Carnivore Food Source Investigation
Check if carnivores are incorrectly accessing food sources
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType
import time

class TrackedFood:
    """Food source with consumption tracking"""
    def __init__(self, original_food):
        self.position = original_food.position
        self.energy_value = original_food.energy_value
        self.regeneration_time = original_food.regeneration_time
        self.current_regen = original_food.current_regen
        self.is_available = original_food.is_available
        self.consumption_log = []
    
    def log_consumption(self, consumer_species, consumer_id):
        """Log who consumed this food"""
        self.consumption_log.append({
            'species': consumer_species,
            'consumer_id': consumer_id,
            'step': len(self.consumption_log)
        })
        print(f"🍽️ FOOD CONSUMED by {consumer_species} (ID: {consumer_id})")

def test_carnivore_food_consumption():
    """Test if carnivores are consuming food sources"""
    print("🔍 Investigating Carnivore Food Consumption")
    print("=" * 60)
    
    # Create environment
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Replace food sources with tracked versions
    tracked_foods = []
    for i, food in enumerate(env.food_sources):
        tracked_food = TrackedFood(food)
        tracked_foods.append(tracked_food)
        env.food_sources[i] = tracked_food
    
    # Get initial counts
    carnivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.CARNIVORE]
    herbivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE]
    
    print(f"🦌 Herbivores: {len(herbivores)}")
    print(f"🐺 Carnivores: {len(carnivores)}")
    print(f"🌾 Food sources: {len(env.food_sources)}")
    print()
    
    # Track specific carnivore
    if carnivores:
        test_carnivore = carnivores[0]
        print(f"📊 Tracking Carnivore ID: {test_carnivore.id}")
        print(f"   Initial Energy: {test_carnivore.energy:.1f}")
        print()
        
        # Place carnivore near food source
        if env.food_sources:
            food = env.food_sources[0]
            test_carnivore.position.x = food.position.x + 1
            test_carnivore.position.y = food.position.y + 1
            print(f"🎯 Placed carnivore near food source")
            print(f"   Carnivore: ({test_carnivore.position.x:.1f}, {test_carnivore.position.y:.1f})")
            print(f"   Food: ({food.position.x:.1f}, {food.position.y:.1f})")
            print(f"   Distance: {test_carnivore.position.distance_to(food.position):.1f}")
            print()
        
        # Run simulation and watch for food consumption
        print("🔄 Running simulation...")
        print("Step | Carnivore Energy | Food Consumed | Notes")
        print("-" * 55)
        
        for step in range(20):
            energy_before = test_carnivore.energy
            foods_available_before = sum(1 for f in env.food_sources if f.is_available)
            
            # Run step
            env.step()
            
            energy_after = test_carnivore.energy
            foods_available_after = sum(1 for f in env.food_sources if f.is_available)
            
            energy_change = energy_after - energy_before
            food_consumed = foods_available_before - foods_available_after
            
            notes = ""
            if energy_change > 0:
                notes += "⚡ ENERGY GAIN! "
            if food_consumed > 0:
                notes += f"🌾 {food_consumed} food consumed "
            if energy_change > 0 and food_consumed > 0:
                notes += "❌ CARNIVORE ATE FOOD!"
            
            print(f"{step:4d} | {energy_after:14.1f} | {food_consumed:13d} | {notes}")
            
            # Check consumption logs
            for i, food in enumerate(tracked_foods):
                if hasattr(food, 'consumption_log') and food.consumption_log:
                    for log_entry in food.consumption_log:
                        if log_entry['step'] == step:
                            if log_entry['species'] == SpeciesType.CARNIVORE:
                                print(f"     ❌ BUG DETECTED: Carnivore consumed food source {i}!")
                                return False
            
            if not test_carnivore.is_alive:
                print("     💀 Carnivore died")
                break
        
        print("\n📈 Analysis:")
        
        # Check all consumption logs
        carnivore_food_consumption = 0
        herbivore_food_consumption = 0
        
        for i, food in enumerate(tracked_foods):
            if hasattr(food, 'consumption_log'):
                for log_entry in food.consumption_log:
                    if log_entry['species'] == SpeciesType.CARNIVORE:
                        carnivore_food_consumption += 1
                    elif log_entry['species'] == SpeciesType.HERBIVORE:
                        herbivore_food_consumption += 1
        
        print(f"   Carnivore food consumption: {carnivore_food_consumption}")
        print(f"   Herbivore food consumption: {herbivore_food_consumption}")
        
        if carnivore_food_consumption > 0:
            print("❌ BUG CONFIRMED: Carnivores are eating food sources!")
            return False
        else:
            print("✅ No bug detected: Carnivores are not eating food sources")
            return True
    
    else:
        print("❌ No carnivores found in environment!")
        return False

def test_feeding_mechanism():
    """Test the feeding mechanism directly"""
    print("\n🔬 Testing Feeding Mechanism Directly")
    print("=" * 45)
    
    env = NeuralEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Find a carnivore
    carnivore = None
    for agent in env.agents:
        if isinstance(agent, NeuralAgent) and agent.species_type == SpeciesType.CARNIVORE:
            carnivore = agent
            break
    
    if not carnivore:
        print("❌ No carnivore found!")
        return False
    
    # Place carnivore next to food
    if env.food_sources:
        food = env.food_sources[0]
        carnivore.position.x = food.position.x + 0.5  # Very close
        carnivore.position.y = food.position.y + 0.5
        
        print(f"🐺 Carnivore energy before: {carnivore.energy:.1f}")
        print(f"🌾 Food available: {food.is_available}")
        print(f"📏 Distance: {carnivore.position.distance_to(food.position):.2f}")
        
        # Call feeding mechanism directly
        old_energy = carnivore.energy
        old_food_available = food.is_available
        
        env._handle_neural_feeding(carnivore)
        
        energy_gained = carnivore.energy - old_energy
        food_consumed = old_food_available and not food.is_available
        
        print(f"🐺 Carnivore energy after: {carnivore.energy:.1f} (change: {energy_gained:+.1f})")
        print(f"🌾 Food available after: {food.is_available}")
        
        if energy_gained > 0 and food_consumed:
            print("❌ BUG: Carnivore gained energy and food was consumed!")
            return False
        elif energy_gained > 0:
            print("❓ Carnivore gained energy but food not consumed - investigate further")
            return False
        elif food_consumed:
            print("❓ Food consumed but carnivore didn't gain energy - investigate further")
            return False
        else:
            print("✅ Correct: No energy gain, no food consumption")
            return True
    
    return False

if __name__ == "__main__":
    try:
        print("🧪 Carnivore Food Source Investigation Suite")
        print("=" * 70)
        
        result1 = test_carnivore_food_consumption()
        result2 = test_feeding_mechanism()
        
        print(f"\n🏁 Final Results:")
        print(f"   Food Consumption Test: {'✅ PASS' if result1 else '❌ FAIL'}")
        print(f"   Direct Feeding Test: {'✅ PASS' if result2 else '❌ FAIL'}")
        
        if result1 and result2:
            print("\n🎉 All tests passed! Carnivores are not eating food sources.")
        else:
            print("\n⚠️ Bug detected! Carnivores may be eating food sources!")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
