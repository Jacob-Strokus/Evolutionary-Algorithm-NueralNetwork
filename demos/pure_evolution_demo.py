"""
Pure Evolution System - Optimized for Best Performance
Based on analysis showing evolution outperforms learning systems
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.neural.neural_agents import NeuralAgent, NeuralEnvironment
from src.evolution.advanced_genetic import AdvancedGeneticAlgorithm, AdvancedEvolutionConfig
from src.core.ecosystem import SpeciesType, Position
import random
import time

class OptimizedEvolutionConfig:
    """Optimized configuration for pure evolution system"""
    def __init__(self):
        # Population parameters
        self.herbivore_population = 15
        self.carnivore_population = 6
        
        # Evolution parameters
        self.elite_percentage = 0.20  # Keep top 20% (increased from 15%)
        self.mutation_rate = 0.15     # Slightly reduced for stability
        self.mutation_strength = 0.25 # Moderate mutations
        self.crossover_rate = 0.8     # High crossover rate
        self.tournament_size = 4      # Increased selection pressure
        
        # Generation parameters
        self.generation_length = 300  # Longer generations for better evaluation
        self.max_generations = 100    # More generations for evolution
        
        # Fitness parameters
        self.diversity_bonus = 0.15   # Increased diversity reward
        self.fitness_sharing = True
        
        # Environment parameters
        self.world_width = 100
        self.world_height = 100

class PureEvolutionSystem:
    """Optimized pure evolution system without learning complications"""
    
    def __init__(self, config: OptimizedEvolutionConfig = None):
        self.config = config or OptimizedEvolutionConfig()
        
        # Create environment
        self.environment = NeuralEnvironment(
            width=self.config.world_width,
            height=self.config.world_height,
            use_neural_agents=True
        )
        
        # Setup advanced genetic algorithm
        genetic_config = AdvancedEvolutionConfig()
        genetic_config.elite_percentage = self.config.elite_percentage
        genetic_config.mutation_rate = self.config.mutation_rate
        genetic_config.mutation_strength = self.config.mutation_strength
        genetic_config.crossover_rate = self.config.crossover_rate
        genetic_config.tournament_size = self.config.tournament_size
        genetic_config.diversity_bonus = self.config.diversity_bonus
        
        self.genetic_algorithm = AdvancedGeneticAlgorithm(genetic_config)
        
        # Evolution tracking
        self.generation = 0
        self.evolution_history = []
        self.best_agents_history = []
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize the starting population"""
        print("🧬 Initializing Pure Evolution Population...")
        
        agents = []
        
        # Create herbivores
        for i in range(self.config.herbivore_population):
            pos = Position(
                random.uniform(10, self.config.world_width - 10),
                random.uniform(10, self.config.world_height - 10)
            )
            agent = NeuralAgent(SpeciesType.HERBIVORE, pos, random.randint(10000, 99999))
            agents.append(agent)
        
        # Create carnivores
        for i in range(self.config.carnivore_population):
            pos = Position(
                random.uniform(10, self.config.world_width - 10),
                random.uniform(10, self.config.world_height - 10)
            )
            agent = NeuralAgent(SpeciesType.CARNIVORE, pos, random.randint(10000, 99999))
            agents.append(agent)
        
        self.environment.agents = agents
        print(f"✅ Created {len(agents)} agents ({self.config.herbivore_population} herbivores, {self.config.carnivore_population} carnivores)")
    
    def run_generation(self, generation_num: int, verbose: bool = True):
        """Run a single generation"""
        if verbose:
            print(f"\n🧬 Generation {generation_num + 1}/{self.config.max_generations}")
            print("-" * 50)
        
        generation_stats = {
            'generation': generation_num,
            'fitness_history': [],
            'population_history': [],
            'diversity_history': [],
            'start_time': time.time()
        }
        
        # Run simulation steps
        for step in range(self.config.generation_length):
            self.environment.step()
            
            # Track statistics every 30 steps
            if step % 30 == 0:
                stats = self.environment.get_neural_stats()
                generation_stats['fitness_history'].append(stats.get('avg_neural_fitness', 0))
                generation_stats['population_history'].append(stats['total_population'])
                
                # Calculate diversity
                diversity = self._calculate_population_diversity()
                generation_stats['diversity_history'].append(diversity)
                
                if verbose and step % 60 == 0:
                    print(f"  Step {step:3d}: "
                          f"Fitness={stats.get('avg_neural_fitness', 0):6.1f}, "
                          f"Pop={stats['total_population']:2d}, "
                          f"Diversity={diversity:5.2f}")
        
        # End of generation statistics
        final_stats = self.environment.get_neural_stats()
        generation_stats['final_fitness'] = final_stats.get('avg_neural_fitness', 0)
        generation_stats['final_population'] = final_stats['total_population']
        generation_stats['final_diversity'] = self._calculate_population_diversity()
        generation_stats['duration'] = time.time() - generation_stats['start_time']
        
        # Record best performers
        best_agents = self._get_best_agents()
        generation_stats['best_agents'] = best_agents
        
        if verbose:
            print(f"  ✅ Final: Fitness={generation_stats['final_fitness']:6.1f}, "
                  f"Pop={generation_stats['final_population']:2d}, "
                  f"Time={generation_stats['duration']:.1f}s")
            
            if best_agents:
                print(f"  🏆 Best: {best_agents[0]['fitness']:.1f} fitness "
                      f"(Gen {best_agents[0]['generation']}, "
                      f"Survival: {best_agents[0]['survival_time']})")
        
        # Evolution to next generation
        if generation_num < self.config.max_generations - 1:
            self._evolve_to_next_generation()
        
        # Store statistics
        self.evolution_history.append(generation_stats)
        return generation_stats
    
    def _calculate_population_diversity(self):
        """Calculate neural network diversity in population"""
        alive_agents = [a for a in self.environment.agents if a.is_alive]
        if len(alive_agents) < 2:
            return 0.0
        
        total_distance = 0
        comparisons = 0
        
        for i, agent1 in enumerate(alive_agents):
            for j, agent2 in enumerate(alive_agents[i+1:], i+1):
                if agent1.species_type == agent2.species_type:
                    # Calculate weight differences
                    dist = np.sum(np.abs(agent1.brain.weights_input_hidden - agent2.brain.weights_input_hidden))
                    dist += np.sum(np.abs(agent1.brain.weights_hidden_output - agent2.brain.weights_hidden_output))
                    total_distance += dist
                    comparisons += 1
        
        return total_distance / max(1, comparisons)
    
    def _get_best_agents(self):
        """Get information about the best performing agents"""
        alive_agents = [a for a in self.environment.agents if a.is_alive]
        alive_agents.sort(key=lambda x: x.brain.fitness_score, reverse=True)
        
        best_agents = []
        for agent in alive_agents[:5]:  # Top 5
            best_agents.append({
                'fitness': agent.brain.fitness_score,
                'species': agent.species_type.value,
                'generation': agent.generation,
                'survival_time': agent.survival_time,
                'offspring_count': agent.offspring_count,
                'food_consumed': agent.lifetime_food_consumed,
                'successful_hunts': agent.lifetime_successful_hunts,
                'energy': agent.energy
            })
        
        return best_agents
    
    def _evolve_to_next_generation(self):
        """Evolve to the next generation using advanced genetic algorithms"""
        print("  🔄 Evolving to next generation...")
        
        current_agents = [a for a in self.environment.agents if a.is_alive]
        
        # Use genetic algorithm for evolution
        next_generation = self.genetic_algorithm.create_next_generation(current_agents)
        
        # Replace environment agents
        self.environment.agents = next_generation
        self.generation += 1
        
        # Log evolution statistics
        genetic_stats = self.genetic_algorithm.get_genetic_stats()
        print(f"    Mutations: {genetic_stats.get('recent_mutations', 0)}, "
              f"Crossovers: {genetic_stats.get('recent_crossovers', 0)}")
    
    def run_evolution(self, verbose: bool = True):
        """Run the complete evolution process"""
        print("🚀 STARTING PURE EVOLUTION SYSTEM")
        print("=" * 60)
        print(f"Generations: {self.config.max_generations}")
        print(f"Generation Length: {self.config.generation_length} steps")
        print(f"Elite Percentage: {self.config.elite_percentage:.0%}")
        print(f"Population: {self.config.herbivore_population + self.config.carnivore_population} agents")
        print()
        
        start_time = time.time()
        
        for generation in range(self.config.max_generations):
            generation_stats = self.run_generation(generation, verbose)
            
            # Early stopping if population dies out
            if generation_stats['final_population'] < 3:
                print("⚠️ Population too small, stopping evolution")
                break
        
        total_time = time.time() - start_time
        
        if verbose:
            print("\n" + "=" * 60)
            print("🎉 EVOLUTION COMPLETE!")
            print(f"Total Time: {total_time:.1f} seconds")
            print(f"Generations: {len(self.evolution_history)}")
            
            if self.evolution_history:
                final_gen = self.evolution_history[-1]
                print(f"Final Fitness: {final_gen['final_fitness']:.1f}")
                print(f"Final Population: {final_gen['final_population']}")
                print(f"Final Diversity: {final_gen['final_diversity']:.2f}")
        
        return self.evolution_history
    
    def plot_evolution_results(self):
        """Plot comprehensive evolution results"""
        if not self.evolution_history:
            print("No evolution data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        generations = range(len(self.evolution_history))
        
        # Final fitness per generation
        final_fitness = [gen['final_fitness'] for gen in self.evolution_history]
        ax1.plot(generations, final_fitness, 'b-', linewidth=2, marker='o')
        ax1.set_title('Evolution Progress: Final Fitness per Generation', fontweight='bold')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Average Fitness')
        ax1.grid(True, alpha=0.3)
        
        # Population survival
        final_population = [gen['final_population'] for gen in self.evolution_history]
        ax2.plot(generations, final_population, 'r-', linewidth=2, marker='s')
        ax2.set_title('Population Survival', fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Population Count')
        ax2.grid(True, alpha=0.3)
        
        # Diversity over time
        final_diversity = [gen['final_diversity'] for gen in self.evolution_history]
        ax3.plot(generations, final_diversity, 'g-', linewidth=2, marker='^')
        ax3.set_title('Population Diversity', fontweight='bold')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Neural Network Diversity')
        ax3.grid(True, alpha=0.3)
        
        # Best agent fitness over generations
        if self.evolution_history:
            best_fitness = []
            for gen in self.evolution_history:
                if gen.get('best_agents'):
                    best_fitness.append(gen['best_agents'][0]['fitness'])
                else:
                    best_fitness.append(0)
            
            ax4.plot(generations, best_fitness, 'purple', linewidth=2, marker='*')
            ax4.set_title('Best Agent Fitness', fontweight='bold')
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Best Fitness Score')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pure_evolution_results.png', dpi=300, bbox_inches='tight')
        print("📊 Evolution results saved to: pure_evolution_results.png")
        plt.show()
    
    def get_evolution_summary(self):
        """Get a summary of evolution results"""
        if not self.evolution_history:
            return {}
        
        initial_fitness = self.evolution_history[0]['final_fitness']
        final_fitness = self.evolution_history[-1]['final_fitness']
        max_fitness = max(gen['final_fitness'] for gen in self.evolution_history)
        
        return {
            'generations_run': len(self.evolution_history),
            'initial_fitness': initial_fitness,
            'final_fitness': final_fitness,
            'max_fitness': max_fitness,
            'improvement': ((final_fitness / max(initial_fitness, 0.1)) - 1) * 100,
            'final_population': self.evolution_history[-1]['final_population'],
            'final_diversity': self.evolution_history[-1]['final_diversity']
        }

def run_pure_evolution_demo():
    """Demonstration of the pure evolution system"""
    print("🧬 PURE EVOLUTION SYSTEM DEMO")
    print("=" * 60)
    
    # Create optimized configuration
    config = OptimizedEvolutionConfig()
    config.max_generations = 20  # Shorter demo
    config.generation_length = 200  # Shorter generations for demo
    
    # Create and run evolution system
    evolution_system = PureEvolutionSystem(config)
    history = evolution_system.run_evolution(verbose=True)
    
    # Show results
    evolution_system.plot_evolution_results()
    summary = evolution_system.get_evolution_summary()
    
    print("\n📊 EVOLUTION SUMMARY:")
    print("-" * 30)
    print(f"Generations: {summary['generations_run']}")
    print(f"Initial Fitness: {summary['initial_fitness']:.1f}")
    print(f"Final Fitness: {summary['final_fitness']:.1f}")
    print(f"Max Fitness: {summary['max_fitness']:.1f}")
    print(f"Improvement: {summary['improvement']:+.1f}%")
    print(f"Final Population: {summary['final_population']}")
    print(f"Final Diversity: {summary['final_diversity']:.2f}")
    
    return evolution_system, history

if __name__ == "__main__":
    evolution_system, history = run_pure_evolution_demo()
    print("\n🎉 Pure Evolution Demo Complete!")
