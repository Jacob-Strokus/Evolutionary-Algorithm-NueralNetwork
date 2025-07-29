#!/usr/bin/env python3
"""
Terminal-based real-time ecosystem visualization
"""
import time
import os
import random
import sys
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def run_terminal_canvas():
    """Run terminal-based real-time visualization"""
    try:
        from src.core.ecosystem import Environment, Position, SpeciesType
        from src.neural.neural_agents import NeuralAgent
        
        print("🎨 Terminal Canvas Demo")
        print("=" * 30)
        
        # Create ecosystem
        env = Environment(width=60, height=30)
        
        # Add some neural agents
        for i in range(8):
            pos = Position(x=random.randint(5, 55), y=random.randint(5, 25))
            herbivore = NeuralAgent(SpeciesType.HERBIVORE, pos, i)
            env.add_agent(herbivore)
        
        for i in range(3):
            pos = Position(x=random.randint(5, 55), y=random.randint(5, 25))
            carnivore = NeuralAgent(SpeciesType.CARNIVORE, pos, i + 100)
            env.add_agent(carnivore)
        
        # Add food
        env.add_food(15)
        
        print("🚀 Running 20 simulation steps...")
        
        for step in range(20):
            print(f"\n--- Step {step + 1} ---")
            
            # Get stats before step
            stats = env.get_neural_stats()
            print(f"Herbivores: {stats['herbivore_count']}, "
                  f"Carnivores: {stats['carnivore_count']}, "
                  f"Food: {len([f for f in env.food_sources if f.is_available])}")
            
            # Step simulation
            env.step()
            
            # Add food occasionally
            if step % 5 == 0:
                env.add_food(3)
            
            time.sleep(0.1)  # Small delay
        
        print("\n✅ Terminal canvas demo completed!")
        
        # Final stats
        final_stats = env.get_neural_stats()
        print(f"Final - Herbivores: {final_stats['herbivore_count']}, "
              f"Carnivores: {final_stats['carnivore_count']}, "
              f"Avg Fitness: {final_stats['avg_neural_fitness']:.2f}")
        
    except Exception as e:
        print(f"❌ Terminal canvas error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_terminal_canvas()