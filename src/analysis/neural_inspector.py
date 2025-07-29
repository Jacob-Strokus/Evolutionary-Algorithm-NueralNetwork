"""
Neural Network Population Inspector
Detailed analysis of individual neural networks in the ecosystem
"""
import numpy as np
import matplotlib.pyplot as plt
from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.neural.neural_network import SensorSystem
from src.core.ecosystem import SpeciesType
import random

class NeuralNetworkInspector:
    """Tool for analyzing individual neural networks in the population"""
    
    def __init__(self, environment: NeuralEnvironment):
        self.environment = environment
        self.neural_agents = [agent for agent in environment.agents if isinstance(agent, NeuralAgent)]
    
    def get_agent_profile(self, agent: NeuralAgent) -> dict:
        """Get comprehensive profile of a neural agent"""
        return {
            'id': agent.id,
            'species': agent.species_type.value,
            'age': agent.age,
            'energy': agent.energy,
            'position': (agent.position.x, agent.position.y),
            'fitness': agent.brain.fitness_score,
            'decisions_made': agent.brain.decisions_made,
            'offspring_count': agent.offspring_count,
            'lifetime_food': agent.lifetime_food_consumed,
            'lifetime_hunts': agent.lifetime_successful_hunts,
            'survival_time': agent.survival_time,
            'can_reproduce': agent.can_reproduce()
        }
    
    def analyze_neural_weights(self, agent: NeuralAgent):
        """Analyze the neural network weights of an agent"""
        brain = agent.brain
        
        print(f"\n🧠 Neural Network Analysis - Agent {agent.id} ({agent.species_type.value})")
        print("=" * 70)
        
        # Basic info
        profile = self.get_agent_profile(agent)
        print(f"📊 Agent Profile:")
        print(f"   Age: {profile['age']} | Energy: {profile['energy']:.1f} | Fitness: {profile['fitness']:.2f}")
        print(f"   Position: ({profile['position'][0]:.1f}, {profile['position'][1]:.1f})")
        print(f"   Decisions Made: {profile['decisions_made']} | Offspring: {profile['offspring_count']}")
        print(f"   Food Consumed: {profile['lifetime_food']} | Hunts: {profile['lifetime_hunts']}")
        
        # Weight statistics
        print(f"\n🔗 Network Architecture:")
        print(f"   Input → Hidden: {brain.weights_input_hidden.shape}")
        print(f"   Hidden → Output: {brain.weights_hidden_output.shape}")
        
        # Input-to-hidden layer analysis
        print(f"\n📥 Input-to-Hidden Weights Analysis:")
        input_weights = brain.weights_input_hidden
        print(f"   Mean: {np.mean(input_weights):.3f} | Std: {np.std(input_weights):.3f}")
        print(f"   Min: {np.min(input_weights):.3f} | Max: {np.max(input_weights):.3f}")
        
        # Hidden-to-output layer analysis
        print(f"\n📤 Hidden-to-Output Weights Analysis:")
        output_weights = brain.weights_hidden_output
        print(f"   Mean: {np.mean(output_weights):.3f} | Std: {np.std(output_weights):.3f}")
        print(f"   Min: {np.min(output_weights):.3f} | Max: {np.max(output_weights):.3f}")
        
        # Bias analysis
        print(f"\n⚖️  Bias Analysis:")
        print(f"   Hidden Biases - Mean: {np.mean(brain.bias_hidden):.3f} | Range: [{np.min(brain.bias_hidden):.3f}, {np.max(brain.bias_hidden):.3f}]")
        print(f"   Output Biases - Mean: {np.mean(brain.bias_output):.3f} | Range: [{np.min(brain.bias_output):.3f}, {np.max(brain.bias_output):.3f}]")
        
        return profile
    
    def test_agent_responses(self, agent: NeuralAgent, num_scenarios: int = 5):
        """Test how an agent responds to different scenarios"""
        print(f"\n🎮 Behavioral Testing - Agent {agent.id}")
        print("=" * 50)
        
        # Create test scenarios
        scenarios = [
            {"name": "High Energy, Food Nearby", "energy": 0.9, "age": 0.2, "food_dist": 0.2, "food_angle": 0.5, "threat_dist": 1.0, "threat_angle": 0.0, "density": 0.3, "can_repro": 1.0},
            {"name": "Low Energy, No Food", "energy": 0.2, "age": 0.5, "food_dist": 1.0, "food_angle": 0.0, "threat_dist": 1.0, "threat_angle": 0.0, "density": 0.1, "can_repro": 0.0},
            {"name": "Threat Nearby", "energy": 0.6, "age": 0.3, "food_dist": 0.8, "food_angle": 0.2, "threat_dist": 0.3, "threat_angle": -0.8, "density": 0.5, "can_repro": 0.0},
            {"name": "Reproduction Ready", "energy": 0.8, "age": 0.4, "food_dist": 0.5, "food_angle": 0.3, "threat_dist": 1.0, "threat_angle": 0.0, "density": 0.2, "can_repro": 1.0},
            {"name": "Crowded Area", "energy": 0.5, "age": 0.6, "food_dist": 0.6, "food_angle": -0.4, "threat_dist": 0.8, "threat_angle": 0.6, "density": 0.9, "can_repro": 0.0}
        ]
        
        for i, scenario in enumerate(scenarios):
            inputs = [
                scenario["energy"], scenario["age"], scenario["food_dist"], scenario["food_angle"],
                scenario["threat_dist"], scenario["threat_angle"], scenario["density"], scenario["can_repro"]
            ]
            
            # Get neural network response
            outputs = agent.brain.forward(np.array(inputs))
            actions = SensorSystem.interpret_network_output(outputs)
            
            print(f"\n📋 Scenario {i+1}: {scenario['name']}")
            print(f"   Inputs: Energy={scenario['energy']:.1f}, Food_dist={scenario['food_dist']:.1f}, Threat_dist={scenario['threat_dist']:.1f}")
            print(f"   🎯 Neural Response:")
            print(f"      Move: ({actions['move_x']:.2f}, {actions['move_y']:.2f})")
            print(f"      Intensity: {actions['intensity']:.2f}")
            print(f"      Reproduce: {'Yes' if actions['should_reproduce'] else 'No'}")
            
            # Interpret behavior
            if abs(actions['move_x']) > 0.3 or abs(actions['move_y']) > 0.3:
                direction = "aggressive" if actions['intensity'] > 0.6 else "cautious"
                print(f"   🤖 Behavior: {direction} movement")
            else:
                print(f"   🤖 Behavior: staying put")
    
    def compare_agents(self, agent1: NeuralAgent, agent2: NeuralAgent):
        """Compare two neural agents"""
        print(f"\n⚖️  Agent Comparison: {agent1.id} vs {agent2.id}")
        print("=" * 60)
        
        profiles = [self.get_agent_profile(agent1), self.get_agent_profile(agent2)]
        
        # Compare basic stats
        print(f"{'Metric':<20} {'Agent ' + str(agent1.id):<15} {'Agent ' + str(agent2.id):<15} {'Winner'}")
        print("-" * 60)
        
        metrics = [
            ('Fitness', 'fitness'), ('Age', 'age'), ('Energy', 'energy'),
            ('Decisions', 'decisions_made'), ('Offspring', 'offspring_count'),
            ('Food Consumed', 'lifetime_food'), ('Hunts', 'lifetime_hunts')
        ]
        
        for metric_name, key in metrics:
            val1, val2 = profiles[0][key], profiles[1][key]
            winner = "Agent " + str(agent1.id) if val1 > val2 else "Agent " + str(agent2.id) if val2 > val1 else "Tie"
            print(f"{metric_name:<20} {val1:<15.1f} {val2:<15.1f} {winner}")
        
        # Compare neural network similarities
        print(f"\n🧠 Neural Network Comparison:")
        weights1 = agent1.brain.weights_input_hidden.flatten()
        weights2 = agent2.brain.weights_input_hidden.flatten()
        
        # Calculate correlation
        correlation = np.corrcoef(weights1, weights2)[0, 1]
        print(f"   Weight Correlation: {correlation:.3f}")
        
        # Calculate weight distance
        weight_distance = np.linalg.norm(weights1 - weights2)
        print(f"   Weight Distance: {weight_distance:.3f}")
        
        if correlation > 0.7:
            print(f"   🔗 Networks are very similar (possible relatives)")
        elif correlation > 0.3:
            print(f"   🔀 Networks are somewhat similar")
        else:
            print(f"   🌟 Networks are quite different")
    
    def find_best_performers(self, top_n: int = 3):
        """Find the best performing agents in the population"""
        if not self.neural_agents:
            print("No neural agents found!")
            return []
        
        # Sort by fitness
        sorted_agents = sorted(self.neural_agents, key=lambda a: a.brain.fitness_score, reverse=True)
        top_agents = sorted_agents[:top_n]
        
        print(f"\n🏆 Top {top_n} Performing Neural Agents")
        print("=" * 60)
        
        for i, agent in enumerate(top_agents):
            profile = self.get_agent_profile(agent)
            print(f"\n🥇 Rank {i+1}: Agent {agent.id} ({agent.species_type.value})")
            print(f"   Fitness: {profile['fitness']:.2f}")
            print(f"   Age: {profile['age']} | Energy: {profile['energy']:.1f}")
            print(f"   Decisions: {profile['decisions_made']} | Offspring: {profile['offspring_count']}")
            
            if agent.species_type == SpeciesType.HERBIVORE:
                print(f"   Food Consumed: {profile['lifetime_food']}")
            else:
                print(f"   Successful Hunts: {profile['lifetime_hunts']}")
        
        return top_agents
    
    def analyze_species_differences(self):
        """Compare neural networks between species"""
        herbivores = [a for a in self.neural_agents if a.species_type == SpeciesType.HERBIVORE]
        carnivores = [a for a in self.neural_agents if a.species_type == SpeciesType.CARNIVORE]
        
        if not herbivores or not carnivores:
            print("Need both herbivores and carnivores for species comparison!")
            return
        
        print(f"\n🦌🐺 Species Neural Network Comparison")
        print("=" * 50)
        
        # Average fitness by species
        herb_fitness = np.mean([a.brain.fitness_score for a in herbivores])
        carn_fitness = np.mean([a.brain.fitness_score for a in carnivores])
        
        print(f"Average Fitness:")
        print(f"   🦌 Herbivores: {herb_fitness:.2f}")
        print(f"   🐺 Carnivores: {carn_fitness:.2f}")
        
        # Average weight magnitudes
        herb_weights = np.mean([np.mean(np.abs(a.brain.weights_input_hidden)) for a in herbivores])
        carn_weights = np.mean([np.mean(np.abs(a.brain.weights_input_hidden)) for a in carnivores])
        
        print(f"\nAverage Weight Magnitudes:")
        print(f"   🦌 Herbivores: {herb_weights:.3f}")
        print(f"   🐺 Carnivores: {carn_weights:.3f}")
        
        # Behavioral tendencies
        print(f"\nSpecies Populations:")
        print(f"   🦌 Herbivores: {len(herbivores)}")
        print(f"   🐺 Carnivores: {len(carnivores)}")

def run_population_analysis():
    """Run comprehensive population analysis"""
    print("🔬 Neural Network Population Analysis")
    print("=" * 60)
    
    # Create environment and let it run for a bit
    env = NeuralEnvironment()
    
    print("🚀 Running simulation to generate neural diversity...")
    for step in range(100):
        env.step()
        if step % 25 == 0:
            stats = env.get_neural_stats()
            print(f"Step {step}: Pop={stats['total_population']}, Fitness={stats.get('avg_neural_fitness', 0):.1f}")
    
    # Create inspector
    inspector = NeuralNetworkInspector(env)
    
    if not inspector.neural_agents:
        print("❌ No neural agents survived to analyze!")
        return
    
    print(f"\n📊 Found {len(inspector.neural_agents)} neural agents to analyze")
    
    # Find best performers
    top_agents = inspector.find_best_performers(3)
    
    # Analyze top performer in detail
    if top_agents:
        print(f"\n🔍 Detailed Analysis of Best Performer:")
        inspector.analyze_neural_weights(top_agents[0])
        inspector.test_agent_responses(top_agents[0])
    
    # Compare species
    inspector.analyze_species_differences()
    
    # Compare top agents if we have multiple
    if len(top_agents) >= 2:
        inspector.compare_agents(top_agents[0], top_agents[1])
    
    return inspector, env

if __name__ == "__main__":
    run_population_analysis()
