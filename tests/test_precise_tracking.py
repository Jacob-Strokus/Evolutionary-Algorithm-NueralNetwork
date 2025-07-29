#!/usr/bin/env python3
"""
Precise Carnivore Food Consumption Test
Track exactly which agents consume food
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType

# Monkey patch to track food consumption
original_consume_food = None
food_consumption_log = []

def tracked_consume_food(self, food_energy):
    """Track which agents consume food"""
    global food_consumption_log
    
    food_consumption_log.append({
        'agent_id': self.id,
        'species': self.species_type,
        'food_energy': food_energy,
        'agent_type': type(self).__name__
    })
    
    print(f"   🍽️ Agent {self.id} ({self.species_type.value}) consumed {food_energy} food energy")
    
    # Call original method
    return original_consume_food(self, food_energy)

def test_precise_carnivore_food_consumption():
    """Test with precise tracking of who eats what"""
    global original_consume_food, food_consumption_log
    
    print("🔍 Precise Carnivore Food Consumption Test")
    print("=" * 60)
    
    # Patch the consume_food method
    original_consume_food = NeuralAgent.consume_food
    NeuralAgent.consume_food = tracked_consume_food
    
    try:
        # Create environment
        env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
        
        # Get agents
        carnivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.CARNIVORE]
        herbivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE]
        
        if not carnivores:
            print("❌ No carnivores found!")
            return False
        
        print(f"🦌 Herbivores: {len(herbivores)}")
        print(f"🐺 Carnivores: {len(carnivores)}")
        print(f"🌾 Food sources: {len(env.food_sources)}")
        print()
        
        # Track a specific carnivore
        test_carnivore = carnivores[0]
        
        # Place carnivore very close to food
        if env.food_sources:
            food = env.food_sources[0]
            test_carnivore.position.x = food.position.x + 0.5  # Very close
            test_carnivore.position.y = food.position.y + 0.5
            
            print(f"🎯 Target Carnivore: ID {test_carnivore.id}")
            print(f"   Position: ({test_carnivore.position.x:.1f}, {test_carnivore.position.y:.1f})")
            print(f"   Food at: ({food.position.x:.1f}, {food.position.y:.1f})")
            print(f"   Distance: {test_carnivore.position.distance_to(food.position):.1f}")
            print()
        
        # Clear log
        food_consumption_log = []
        
        # Run simulation
        print("🔄 Running detailed simulation...")
        print()
        
        for step in range(15):
            print(f"--- Step {step + 1} ---")
            
            # Record state before
            energy_before = test_carnivore.energy
            
            # Run step
            env.step()
            
            # Record state after
            energy_after = test_carnivore.energy
            energy_change = energy_after - energy_before
            
            print(f"Carnivore {test_carnivore.id}: {energy_before:.1f} -> {energy_after:.1f} ({energy_change:+.1f})")
            
            if not test_carnivore.is_alive:
                print("💀 Carnivore died!")
                break
            
            print()
        
        print("\n📊 Consumption Analysis:")
        print("-" * 40)
        
        carnivore_consumptions = [log for log in food_consumption_log if log['species'] == SpeciesType.CARNIVORE]
        herbivore_consumptions = [log for log in food_consumption_log if log['species'] == SpeciesType.HERBIVORE]
        
        print(f"Total food consumptions: {len(food_consumption_log)}")
        print(f"Carnivore consumptions: {len(carnivore_consumptions)}")
        print(f"Herbivore consumptions: {len(herbivore_consumptions)}")
        print()
        
        if carnivore_consumptions:
            print("🚨 CARNIVORE FOOD CONSUMPTION DETECTED:")
            for consumption in carnivore_consumptions:
                print(f"   Agent {consumption['agent_id']}: {consumption['food_energy']} energy")
            return False
        else:
            print("✅ No carnivore food consumption detected")
            
        if herbivore_consumptions:
            print(f"✅ Herbivore food consumption: {len(herbivore_consumptions)} instances")
        
        return True
        
    finally:
        # Restore original method
        if original_consume_food:
            NeuralAgent.consume_food = original_consume_food

if __name__ == "__main__":
    try:
        success = test_precise_carnivore_food_consumption()
        if success:
            print("\n🎉 Test PASSED: Carnivores are not eating food!")
        else:
            print("\n❌ Test FAILED: Carnivores are eating food sources!")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
