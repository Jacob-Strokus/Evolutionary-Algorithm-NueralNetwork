"""
Quick Neural Agent Analysis
"""
from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.neural.neural_network import SensorSystem
import numpy as np

def quick_neural_analysis():
    """Quick analysis of neural agents"""
    print("🔬 Quick Neural Analysis")
    print("=" * 40)
    
    # Create and run environment
    env = NeuralEnvironment()
    for step in range(25):  # Short run to keep more agents alive
        env.step()
    
    # Get neural agents
    neural_agents = [agent for agent in env.agents if isinstance(agent, NeuralAgent) and agent.is_alive]
    
    print(f"Found {len(neural_agents)} living neural agents")
    
    # Sort by fitness
    neural_agents.sort(key=lambda a: a.brain.fitness_score, reverse=True)
    
    # Examine top 3 agents
    for i, agent in enumerate(neural_agents[:3]):
        print(f"\n🎯 Agent {i+1}: ID={agent.id} ({agent.species_type.value})")
        print(f"   Fitness: {agent.brain.fitness_score:.1f}")
        print(f"   Energy: {agent.energy:.0f}")
        print(f"   Age: {agent.age}")
        
        # Test with a standard input
        test_input = [0.5, 0.3, 0.4, 0.2, 0.7, 0.1, 0.3, 0.0]
        output = agent.brain.forward(np.array(test_input))
        actions = SensorSystem.interpret_network_output(output)
        
        print(f"   Test Move: ({actions['move_x']:.2f}, {actions['move_y']:.2f})")
        print(f"   Intensity: {actions['intensity']:.2f}")
        
        # Show weight stats
        w_mean = np.mean(agent.brain.weights_input_hidden)
        w_std = np.std(agent.brain.weights_input_hidden)
        print(f"   Weights: μ={w_mean:.3f}, σ={w_std:.3f}")

if __name__ == "__main__":
    for run in range(3):
        print(f"\n🔄 Run {run + 1}")
        quick_neural_analysis()
        print("-" * 40)
