#!/usr/bin/env python3
"""
Simple test to check if carnivores eat food
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType

def test_carnivore_food_bug():
    """Simple test to see if carnivores eat food"""
    print("🧪 Simple Carnivore Food Test")
    print("=" * 40)
    
    env = NeuralEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Find carnivore
    carnivore = None
    for agent in env.agents:
        if isinstance(agent, NeuralAgent) and agent.species_type == SpeciesType.CARNIVORE:
            carnivore = agent
            break
    
    if not carnivore:
        print("No carnivore found!")
        return
    
    print(f"Found carnivore {carnivore.id}")
    print(f"Initial energy: {carnivore.energy}")
    
    # Place next to food
    if env.food_sources:
        food = env.food_sources[0]
        carnivore.position.x = food.position.x + 0.1
        carnivore.position.y = food.position.y + 0.1
        print(f"Placed carnivore at distance {carnivore.position.distance_to(food.position):.2f} from food")
        print(f"Food available: {food.is_available}")
    
    # Try to call consume_food directly
    try:
        print("\nTesting direct consume_food call...")
        energy_before = carnivore.energy
        carnivore.consume_food(40)
        energy_after = carnivore.energy
        
        if energy_after > energy_before:
            print(f"❌ BUG CONFIRMED: Carnivore gained {energy_after - energy_before} energy from food!")
        else:
            print("✅ Direct call blocked - carnivore didn't gain energy")
    except Exception as e:
        print(f"Exception during consume_food: {e}")
    
    return True

if __name__ == "__main__":
    test_carnivore_food_bug()
