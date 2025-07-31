"""
Enhanced Main Evolution System - Optimized Pure Evolution
Based on performance analysis showing evolution outperforms learning
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment
from src.evolution.advanced_genetic import AdvancedGeneticAlgorithm, AdvancedEvolutionConfig
from src.visualization.neural_visualizer import visualize_neural_ecosystem
import time

class EnhancedEvolutionSystem:
    """Main system optimized for pure evolution performance"""
    
    def __init__(self, width=120, height=120):
        print("🧬 Initializing Enhanced Evolution System...")
        
        # Create optimized environment
        self.environment = NeuralEnvironment(width=width, height=height, use_neural_agents=True)
        
        # Setup advanced genetic algorithm with optimized parameters
        genetic_config = AdvancedEvolutionConfig()
        genetic_config.elite_percentage = 0.20      # Keep top 20%
        genetic_config.mutation_rate = 0.15         # Moderate mutations
        genetic_config.mutation_strength = 0.25     # Balanced strength
        genetic_config.crossover_rate = 0.8         # High crossover
        genetic_config.tournament_size = 4          # Strong selection pressure
        genetic_config.diversity_bonus = 0.15       # Encourage diversity
        genetic_config.generation_length = 400      # Longer evaluation
        
        self.genetic_algorithm = AdvancedGeneticAlgorithm(genetic_config)
        
        # Evolution tracking
        self.generation = 0
        self.total_steps = 0
        self.evolution_stats = []
        
        print("✅ Enhanced Evolution System Ready!")
    
    def run_evolution_cycle(self, steps_per_generation=400, max_generations=50, 
                           show_visualization=True, verbose=True):
        """Run multiple generations of evolution"""
        
        print(f"\n🚀 STARTING EVOLUTION CYCLE")
        print(f"Generations: {max_generations}, Steps per generation: {steps_per_generation}")
        print("=" * 60)
        
        start_time = time.time()
        
        for gen in range(max_generations):
            gen_start_time = time.time()
            
            if verbose:
                print(f"\n🧬 Generation {gen + 1}/{max_generations}")
                print("-" * 40)
            
            # Run generation
            generation_stats = self.run_single_generation(
                steps_per_generation, 
                show_visualization and gen % 5 == 0,  # Show viz every 5th generation
                verbose
            )
            
            generation_stats['generation'] = gen + 1
            generation_stats['duration'] = time.time() - gen_start_time
            self.evolution_stats.append(generation_stats)
            
            if verbose:
                print(f"  ⏱️  Generation time: {generation_stats['duration']:.1f}s")
                print(f"  📊 Best fitness: {generation_stats['best_fitness']:.1f}")
                print(f"  👥 Population: {generation_stats['final_population']}")
            
            # Check for extinction
            if generation_stats['final_population'] < 5:
                print("⚠️ Population too small, ending evolution")
                break
            
            # Evolve to next generation (except for last generation)
            if gen < max_generations - 1:
                self.evolve_to_next_generation()
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 EVOLUTION CYCLE COMPLETE!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Generations completed: {len(self.evolution_stats)}")
        
        if self.evolution_stats:
            final_stats = self.evolution_stats[-1]
            initial_stats = self.evolution_stats[0]
            improvement = ((final_stats['best_fitness'] / max(initial_stats['best_fitness'], 0.1)) - 1) * 100
            
            print(f"Initial best fitness: {initial_stats['best_fitness']:.1f}")
            print(f"Final best fitness: {final_stats['best_fitness']:.1f}")
            print(f"Improvement: {improvement:+.1f}%")
        
        return self.evolution_stats
    
    def run_single_generation(self, steps=400, show_visualization=False, verbose=True):
        """Run a single generation and collect statistics"""
        
        generation_stats = {
            'steps': 0,
            'fitness_progression': [],
            'population_progression': [],
            'best_fitness': 0,
            'average_fitness': 0,
            'final_population': 0,
            'genetic_operations': 0
        }
        
        # Run simulation steps
        for step in range(steps):
            self.environment.step()
            self.total_steps += 1
            generation_stats['steps'] += 1
            
            # Collect statistics every 40 steps
            if step % 40 == 0:
                stats = self.environment.get_neural_stats()
                generation_stats['fitness_progression'].append(stats.get('avg_neural_fitness', 0))
                generation_stats['population_progression'].append(stats.get('total_population', 0))
                
                if verbose and step % 80 == 0:
                    print(f"    Step {step:3d}: "
                          f"Fitness={stats.get('avg_neural_fitness', 0):6.1f}, "
                          f"Population={stats.get('total_population', 0):2d}")
            
            # Show visualization periodically
            if show_visualization and step % 100 == 0:
                visualize_neural_ecosystem(self.environment)
        
        # Final statistics
        final_stats = self.environment.get_neural_stats()
        generation_stats['best_fitness'] = final_stats.get('max_neural_fitness', 0)
        generation_stats['average_fitness'] = final_stats.get('avg_neural_fitness', 0)
        generation_stats['final_population'] = final_stats.get('total_population', 0)
        
        # Get genetic operations statistics
        genetic_stats = self.genetic_algorithm.get_genetic_stats()
        generation_stats['genetic_operations'] = (
            genetic_stats.get('recent_mutations', 0) + 
            genetic_stats.get('recent_crossovers', 0)
        )
        
        return generation_stats
    
    def evolve_to_next_generation(self):
        """Evolve the population to the next generation"""
        print("    🔄 Evolving to next generation...")
        
        current_agents = [a for a in self.environment.agents if a.is_alive]
        
        # Calculate target populations (maintain ratio)
        total_alive = len(current_agents)
        herbivore_count = len([a for a in current_agents if a.species_type.value == 'herbivore'])
        carnivore_count = total_alive - herbivore_count
        
        # Maintain roughly 3:1 herbivore to carnivore ratio
        target_total = max(20, min(30, total_alive + 2))  # Gradually grow population
        target_herbivores = int(target_total * 0.75)
        target_carnivores = target_total - target_herbivores
        
        # Perform evolution
        next_generation = self.genetic_algorithm.create_next_generation(current_agents)
        
        # Update environment
        self.environment.agents = next_generation
        self.generation += 1
        
        print(f"      New population: {len(next_generation)} agents "
              f"({target_herbivores}H, {target_carnivores}C)")
    
    def get_evolution_summary(self):
        """Get comprehensive evolution summary"""
        if not self.evolution_stats:
            return {}
        
        best_fitnesses = [gen['best_fitness'] for gen in self.evolution_stats]
        avg_fitnesses = [gen['average_fitness'] for gen in self.evolution_stats]
        populations = [gen['final_population'] for gen in self.evolution_stats]
        
        return {
            'total_generations': len(self.evolution_stats),
            'total_steps': self.total_steps,
            'initial_best_fitness': best_fitnesses[0] if best_fitnesses else 0,
            'final_best_fitness': best_fitnesses[-1] if best_fitnesses else 0,
            'max_fitness_achieved': max(best_fitnesses) if best_fitnesses else 0,
            'average_final_fitness': avg_fitnesses[-1] if avg_fitnesses else 0,
            'final_population': populations[-1] if populations else 0,
            'population_stability': len([p for p in populations if p >= 10]) / len(populations) if populations else 0,
            'fitness_improvement': ((best_fitnesses[-1] / max(best_fitnesses[0], 0.1)) - 1) * 100 if best_fitnesses else 0
        }
    
    def save_evolution_data(self, filename="evolution_data.json"):
        """Save evolution data to file"""
        import json
        
        data = {
            'evolution_stats': self.evolution_stats,
            'summary': self.get_evolution_summary(),
            'configuration': {
                'environment_size': f"{self.environment.width}x{self.environment.height}",
                'genetic_config': self.genetic_algorithm.config.__dict__ if hasattr(self.genetic_algorithm.config, '__dict__') else str(self.genetic_algorithm.config)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Evolution data saved to: {filename}")

def main():
    """Main function to run the enhanced evolution system"""
    print("🧬 ENHANCED EVOLUTION SYSTEM")
    print("=" * 50)
    
    # Create system
    system = EnhancedEvolutionSystem(width=100, height=100)
    
    # Run evolution
    results = system.run_evolution_cycle(
        steps_per_generation=300,
        max_generations=10,
        show_visualization=False,  # Set to True to see neural network visualization
        verbose=True
    )
    
    # Show summary
    summary = system.get_evolution_summary()
    print("\n📊 EVOLUTION SUMMARY:")
    print("-" * 30)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Save data
    system.save_evolution_data("enhanced_evolution_results.json")
    
    return system, results

if __name__ == "__main__":
    system, results = main()
    print("\n🎉 Enhanced Evolution System Complete!")
