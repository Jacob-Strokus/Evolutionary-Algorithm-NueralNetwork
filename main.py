#!/usr/bin/env python3
"""
Neural Ecosystem Simulation - Main Entry Point
==============================================

Main launcher for the neural network ecosystem simulation with generation tracking
and web-based visualization. This is the primary entry point for running the
complete neural ecosystem with all features enabled.

Usage:
    python main.py              # Run standard neural simulation
    python main.py --web        # Run with web interface
    python main.py --analysis   # Run with detailed analysis
    python main.py --extended   # Run extended 1500-step simulation
"""

import sys
import os
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.analysis.neural_inspector import NeuralNetworkInspector
from src.core.ecosystem import SpeciesType

class SimpleEcosystemWrapper:
    """Simple wrapper for the environment to work with the web server"""
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        self.fitness_history_herb = []
        self.fitness_history_carn = []
        self.population_history = []
    
    def get_agent_data(self):
        """Get agent data in the format expected by the web server"""
        herbivores_data = {'x': [], 'y': [], 'fitness': [], 'ids': []}
        carnivores_data = {'x': [], 'y': [], 'fitness': [], 'ids': []}
        food_data = {'x': [], 'y': []}
        
        # Process agents
        for agent in self.env.agents:
            if agent.species_type == SpeciesType.HERBIVORE:
                herbivores_data['x'].append(agent.position.x)
                herbivores_data['y'].append(agent.position.y)
                herbivores_data['fitness'].append(getattr(agent.brain, 'fitness_score', 0) if hasattr(agent, 'brain') else 0)
                herbivores_data['ids'].append(getattr(agent, 'agent_id', id(agent)))
            else:  # CARNIVORE
                carnivores_data['x'].append(agent.position.x)
                carnivores_data['y'].append(agent.position.y)
                carnivores_data['fitness'].append(getattr(agent.brain, 'fitness_score', 0) if hasattr(agent, 'brain') else 0)
                carnivores_data['ids'].append(getattr(agent, 'agent_id', id(agent)))
        
        # Process food sources
        for food in self.env.food_sources:
            if food.is_available:  # Only show available food
                food_data['x'].append(food.position.x)
                food_data['y'].append(food.position.y)
        
        return {
            'herbivores': herbivores_data,
            'carnivores': carnivores_data,
            'food': food_data
        }
    
    def update_step(self):
        """Update step count and run environment update"""
        self.step_count += 1
        self.env.update()

def run_standard_simulation(steps=500):
    """Run a standard neural ecosystem simulation."""
    print("🧠 Starting Neural Ecosystem Simulation")
    print("=" * 50)
    
    # Create neural environment with generation tracking
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    print(f"🦌 Initial Herbivores: {len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE])}")
    print(f"🐺 Initial Carnivores: {len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE])}")
    print(f"🌱 Food Sources: {len(env.food_sources)}")
    print("\n🔄 Running simulation...")
    
    # Run simulation
    for step in range(steps):
        if step % 100 == 0:
            herbivores = len([a for a in env.agents if a.species_type == SpeciesType.HERBIVORE])
            carnivores = len([a for a in env.agents if a.species_type == SpeciesType.CARNIVORE])
            generations = set(a.generation for a in env.agents if hasattr(a, 'generation'))
            
            print(f"Step {step}: 🦌 {herbivores} herbivores, 🐺 {carnivores} carnivores, 🧬 Generations: {sorted(generations)}")
        
        env.update()
        
        # Check for extinction
        if len(env.agents) == 0:
            print(f"💀 Ecosystem collapse at step {step}")
            break
    
    # Final statistics
    print("\n📊 Final Statistics:")
    final_herbivores = [a for a in env.agents if a.species_type == SpeciesType.HERBIVORE]
    final_carnivores = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE]
    
    print(f"🦌 Final Herbivores: {len(final_herbivores)}")
    print(f"🐺 Final Carnivores: {len(final_carnivores)}")
    
    if env.agents:
        generations = [a.generation for a in env.agents if hasattr(a, 'generation')]
        if generations:
            print(f"🧬 Generation Range: {min(generations)} - {max(generations)}")
            print(f"📈 Average Generation: {sum(generations) / len(generations):.1f}")
        
        # Show top performers
        if final_herbivores:
            top_herbivore = max(final_herbivores, key=lambda a: a.energy)
            print(f"🏆 Top Herbivore: Gen {getattr(top_herbivore, 'generation', '?')}, Energy: {top_herbivore.energy:.1f}")
        
        if final_carnivores:
            top_carnivore = max(final_carnivores, key=lambda a: a.energy)
            print(f"🏆 Top Carnivore: Gen {getattr(top_carnivore, 'generation', '?')}, Energy: {top_carnivore.energy:.1f}")

def run_web_simulation(steps=1000):
    """Run simulation with web-based visualization."""
    print("🌐 Starting Neural Ecosystem with Web Interface")
    print("=" * 50)
    
    # Import web server only when needed
    from src.visualization.realtime_web_server import EcosystemWebServer
    
    # Create environment
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Create simple wrapper for web server
    canvas = SimpleEcosystemWrapper(env)
    
    # Create web server
    web_server = EcosystemWebServer(canvas)
    
    print("🚀 Starting web server...")
    print("📱 Open your browser to: http://localhost:5000")
    print("🔍 Click on agents to inspect their neural networks!")
    print("⚡ Press Ctrl+C to stop the simulation")
    
    try:
        # Start the web server (this will run the simulation loop)
        web_server.start_server()
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")

def run_analysis_simulation(steps=500):
    """Run simulation with detailed neural network analysis."""
    print("🔬 Starting Neural Ecosystem Analysis")
    print("=" * 50)
    
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    inspector = NeuralNetworkInspector()
    
    print("🧪 Running detailed analysis simulation...")
    
    # Track generation evolution
    generation_history = []
    
    for step in range(steps):
        if step % 50 == 0:
            # Analyze current state
            generations = [a.generation for a in env.agents if hasattr(a, 'generation')]
            if generations:
                generation_history.append({
                    'step': step,
                    'avg_generation': sum(generations) / len(generations),
                    'max_generation': max(generations),
                    'agent_count': len(env.agents)
                })
            
            if step % 100 == 0:
                print(f"Step {step}: Analyzing neural networks...")
                
                # Sample a few agents for detailed analysis
                sample_agents = env.agents[:3] if len(env.agents) >= 3 else env.agents
                for agent in sample_agents:
                    if hasattr(agent, 'neural_network'):
                        analysis = inspector.analyze_network(agent.neural_network)
                        print(f"  Agent {agent.agent_id}: Gen {getattr(agent, 'generation', '?')}, "
                              f"Complexity: {analysis.get('complexity', 'N/A')}")
        
        env.update()
    
    # Final analysis report
    print("\n📈 Generation Evolution Analysis:")
    for record in generation_history[-5:]:  # Show last 5 records
        print(f"  Step {record['step']}: Avg Gen {record['avg_generation']:.1f}, "
              f"Max Gen {record['max_generation']}, Agents: {record['agent_count']}")

def main():
    """Main entry point with command line argument parsing."""
    print("🚀 Main function started")
    parser = argparse.ArgumentParser(description='Neural Ecosystem Simulation')
    parser.add_argument('--web', action='store_true', help='Run with web interface')
    parser.add_argument('--analysis', action='store_true', help='Run with detailed analysis')
    parser.add_argument('--extended', action='store_true', help='Run extended 1500-step simulation')
    parser.add_argument('--steps', type=int, default=500, help='Number of simulation steps')
    
    args = parser.parse_args()
    print(f"📝 Parsed arguments: {args}")
    
    # Determine simulation type
    if args.web:
        print("🌐 Starting web simulation...")
        run_web_simulation(args.steps if args.steps != 500 else 1000)
    elif args.analysis:
        print("🔬 Starting analysis simulation...")
        run_analysis_simulation(args.steps)
    elif args.extended:
        print("⏳ Starting extended simulation...")
        run_standard_simulation(1500)
    else:
        print("🧠 Starting standard simulation...")
        run_standard_simulation(args.steps)

if __name__ == "__main__":
    main()
