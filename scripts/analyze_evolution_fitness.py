"""
Quick Evolution Fitness Analysis
Tests just the evolution system to understand the fitness algorithm
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.neural.neural_agents import NeuralAgent, NeuralEnvironment
from src.core.ecosystem import SpeciesType, Position
import random

def analyze_evolution_fitness():
    """Analyze how the evolution fitness system works"""
    print("🧬 EVOLUTION FITNESS ALGORITHM ANALYSIS")
    print("=" * 60)
    
    # Create a simple environment
    env = NeuralEnvironment(width=80, height=80, use_neural_agents=True)
    
    # Create some test agents
    test_agents = []
    for i in range(5):
        pos = Position(random.uniform(10, 70), random.uniform(10, 70))
        agent = NeuralAgent(SpeciesType.HERBIVORE, pos, random.randint(10000, 99999))
        test_agents.append(agent)
        env.agents.append(agent)
    
    print("📊 Initial agent fitness scores:")
    for i, agent in enumerate(test_agents):
        print(f"  Agent {i+1}: {agent.brain.fitness_score:.2f}")
    
    print("\n🔄 Running simulation for 50 steps...")
    
    # Track fitness evolution
    fitness_history = [[] for _ in test_agents]
    
    for step in range(50):
        env.step()
        
        # Track fitness for each agent
        for i, agent in enumerate(test_agents):
            if agent.is_alive:
                fitness_history[i].append(agent.brain.fitness_score)
            else:
                fitness_history[i].append(0)  # Dead agent
        
        if step % 10 == 0:
            print(f"\n  Step {step}:")
            for i, agent in enumerate(test_agents):
                if agent.is_alive:
                    print(f"    Agent {i+1}: Fitness={agent.brain.fitness_score:6.1f}, "
                          f"Energy={agent.energy:3.0f}, Survival={agent.survival_time:3d}, "
                          f"Food={agent.lifetime_food_consumed:2d}, Offspring={agent.offspring_count}")
                else:
                    print(f"    Agent {i+1}: DEAD")
    
    print("\n📈 FITNESS ALGORITHM BREAKDOWN:")
    print("=" * 50)
    
    # Analyze the fitness components for surviving agents
    for i, agent in enumerate(test_agents):
        if agent.is_alive:
            print(f"\n🤖 Agent {i+1} (ID: {agent.id}):")
            
            # Calculate fitness components (from neural_agents.py update_fitness method)
            survival_fitness = agent.survival_time * 0.1
            energy_fitness = (agent.energy / agent.max_energy) * 10
            species_bonus = agent.lifetime_food_consumed * 2  # Herbivore bonus
            reproduction_fitness = agent.offspring_count * 25
            
            total_calculated = survival_fitness + energy_fitness + species_bonus + reproduction_fitness
            
            print(f"  Survival Fitness:    {survival_fitness:6.1f} (time: {agent.survival_time} × 0.1)")
            print(f"  Energy Fitness:      {energy_fitness:6.1f} (energy: {agent.energy}/{agent.max_energy})")
            print(f"  Species Bonus:       {species_bonus:6.1f} (food: {agent.lifetime_food_consumed} × 2)")
            print(f"  Reproduction Bonus:  {reproduction_fitness:6.1f} (offspring: {agent.offspring_count} × 25)")
            print(f"  Calculated Total:    {total_calculated:6.1f}")
            print(f"  Actual Brain Score:  {agent.brain.fitness_score:6.1f}")
            print(f"  Momentum Effect:     {agent.brain.fitness_score - total_calculated:6.1f}")
    
    print("\n🎯 KEY INSIGHTS:")
    print("-" * 30)
    print("1. Survival Time: +0.1 per step alive")
    print("2. Energy Management: +10 points at full energy")
    print("3. Food Consumption (Herbivores): +2 per food item")
    print("4. Successful Hunts (Carnivores): +15 per hunt")
    print("5. Reproduction: +25 per offspring")
    print("6. Momentum: 90% previous + 10% current (smoothing)")
    
    return test_agents

def compare_fitness_systems():
    """Compare evolution vs learning fitness approaches"""
    print("\n" + "=" * 60)
    print("🆚 EVOLUTION vs LEARNING FITNESS COMPARISON")
    print("=" * 60)
    
    print("\n🧬 EVOLUTION FITNESS STRENGTHS:")
    print("✅ Long-term optimization across generations")
    print("✅ Population-level selection pressure")
    print("✅ Elite preservation (top 15% always survive)")
    print("✅ Tournament selection ensures quality breeding")
    print("✅ Diversity bonuses prevent homogeneity")
    print("✅ Cumulative fitness over entire lifetime")
    print("✅ Simple, stable reward structure")
    
    print("\n🎓 LEARNING FITNESS CHALLENGES:")
    print("❌ Real-time reward calculation complexity")
    print("❌ Immediate penalties can destabilize weights")
    print("❌ Boundary penalties too harsh (-10 to -50)")
    print("❌ Learning rate too aggressive (weight explosion)")
    print("❌ No population-level selection pressure")
    print("❌ Individual optimization without global context")
    print("❌ Reward structure sensitive to tuning")
    
    print("\n💡 WHY EVOLUTION OUTPERFORMS LEARNING:")
    print("1. Stable fitness accumulation vs volatile real-time rewards")
    print("2. Population diversity vs individual learning instability")
    print("3. Long-term selection pressure vs short-term feedback")
    print("4. Elite preservation vs random weight updates")
    print("5. Tournament selection vs gradient-based updates")

if __name__ == "__main__":
    test_agents = analyze_evolution_fitness()
    compare_fitness_systems()
    print("\n🎉 Evolution fitness analysis complete!")
