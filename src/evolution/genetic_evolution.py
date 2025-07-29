"""
Phase 3: Evolution & Genetic Algorithms
Implements population-wide selection pressure, elitism, and generational evolution
"""
import numpy as np
import random
import math
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from src.neural.neural_agents import NeuralAgent, NeuralEnvironment
from src.neural.neural_network import NeuralNetwork, NeuralNetworkConfig
from src.core.ecosystem import SpeciesType, Position
import json
import copy

@dataclass
class EvolutionConfig:
    """Configuration for genetic algorithm parameters"""
    population_size: int = 40
    elite_percentage: float = 0.15  # Top 15% always survive
    mutation_rate: float = 0.2
    mutation_strength: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 3
    generation_length: int = 1000  # Steps per generation
    max_generations: int = 50
    fitness_sharing: bool = True
    diversity_bonus: float = 0.1

class GeneticAlgorithm:
    """Manages genetic algorithm operations for neural agent evolution"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generation = 0
        self.evolution_history = []
        self.species_stats = {"herbivore": [], "carnivore": []}
        
        # Track best performers across generations
        self.hall_of_fame = {"herbivore": [], "carnivore": []}
        
        # Track genetic operations for visualization
        self.genetic_events = {
            'mutations': [],
            'crossovers': [],
            'recent_mutations': 0,
            'recent_crossovers': 0,
            'total_mutations': 0,
            'total_crossovers': 0
        }
        
    def evaluate_population(self, agents: List[NeuralAgent]) -> Dict:
        """Evaluate fitness of entire population"""
        fitness_stats = {
            "herbivore": {"fitness": [], "agents": []},
            "carnivore": {"fitness": [], "agents": []}
        }
        
        for agent in agents:
            if not agent.is_alive:
                continue
                
            species_key = "herbivore" if agent.species_type == SpeciesType.HERBIVORE else "carnivore"
            fitness_stats[species_key]["fitness"].append(agent.brain.fitness_score)
            fitness_stats[species_key]["agents"].append(agent)
        
        # Calculate statistics
        for species in ["herbivore", "carnivore"]:
            if fitness_stats[species]["fitness"]:
                fitness_values = fitness_stats[species]["fitness"]
                fitness_stats[species]["mean"] = np.mean(fitness_values)
                fitness_stats[species]["std"] = np.std(fitness_values)
                fitness_stats[species]["max"] = np.max(fitness_values)
                fitness_stats[species]["min"] = np.min(fitness_values)
            else:
                fitness_stats[species].update({"mean": 0, "std": 0, "max": 0, "min": 0})
        
        return fitness_stats
    
    def selection_tournament(self, agents: List[NeuralAgent], species_type: SpeciesType) -> NeuralAgent:
        """Tournament selection for breeding"""
        # Filter agents by species
        species_agents = [a for a in agents if a.species_type == species_type and a.is_alive]
        
        if len(species_agents) < self.config.tournament_size:
            return random.choice(species_agents) if species_agents else None
        
        # Select random tournament participants
        tournament = random.sample(species_agents, self.config.tournament_size)
        
        # Return agent with highest fitness
        return max(tournament, key=lambda x: x.brain.fitness_score)
    
    def calculate_diversity_bonus(self, agent: NeuralAgent, population: List[NeuralAgent]) -> float:
        """Calculate diversity bonus based on neural network uniqueness"""
        if not self.config.fitness_sharing:
            return 0.0
        
        same_species_agents = [a for a in population 
                              if a.species_type == agent.species_type and a != agent and a.is_alive]
        
        if not same_species_agents:
            return self.config.diversity_bonus
        
        # Calculate average weight distance to other agents
        distances = []
        for other in same_species_agents:
            # Compare neural network weights
            dist = np.sum(np.abs(agent.brain.weights_input_hidden - other.brain.weights_input_hidden))
            dist += np.sum(np.abs(agent.brain.weights_hidden_output - other.brain.weights_hidden_output))
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        # Normalize diversity bonus (higher distance = more unique = higher bonus)
        diversity_bonus = min(self.config.diversity_bonus, avg_distance / 100.0)
        
        return diversity_bonus
    
    def apply_elitism(self, agents: List[NeuralAgent], species_type: SpeciesType) -> List[NeuralAgent]:
        """Select elite agents that automatically survive to next generation"""
        species_agents = [a for a in agents if a.species_type == species_type and a.is_alive]
        species_agents.sort(key=lambda x: x.brain.fitness_score, reverse=True)
        
        elite_count = max(1, int(len(species_agents) * self.config.elite_percentage))
        elite_agents = species_agents[:elite_count]
        
        # Update hall of fame
        species_key = "herbivore" if species_type == SpeciesType.HERBIVORE else "carnivore"
        if elite_agents:
            best_agent = elite_agents[0]
            self.hall_of_fame[species_key].append({
                "generation": self.generation,
                "fitness": best_agent.brain.fitness_score,
                "agent_id": best_agent.id,  # Fixed: use 'id' not 'agent_id'
                "survival_time": best_agent.survival_time,
                "offspring_count": best_agent.offspring_count
            })
        
        return elite_agents
    
    def crossover_networks(self, parent1: NeuralAgent, parent2: NeuralAgent) -> NeuralNetwork:
        """Advanced crossover between two neural networks"""
        if random.random() > self.config.crossover_rate:
            # No crossover, return mutated copy of parent1
            offspring_brain = parent1.brain.copy()
            offspring_brain.mutate()
            
            # Track mutation event
            self.genetic_events['mutations'].append({
                'timestamp': time.time(),
                'type': 'mutation',
                'parent_species': parent1.species_type.value,
                'parent_fitness': parent1.brain.fitness_score,
                'generation': self.generation
            })
            self.genetic_events['recent_mutations'] += 1
            self.genetic_events['total_mutations'] += 1
            
            return offspring_brain
        
        # Track crossover event
        self.genetic_events['crossovers'].append({
            'timestamp': time.time(),
            'type': 'crossover',
            'parent1_species': parent1.species_type.value,
            'parent2_species': parent2.species_type.value,
            'parent1_fitness': parent1.brain.fitness_score,
            'parent2_fitness': parent2.brain.fitness_score,
            'generation': self.generation
        })
        self.genetic_events['recent_crossovers'] += 1
        self.genetic_events['total_crossovers'] += 1
        
        # Create offspring network
        config = parent1.brain.config
        offspring_brain = NeuralNetwork(config)
        
        # Multi-point crossover for different weight matrices
        
        # Input-to-hidden weights crossover
        for i in range(offspring_brain.weights_input_hidden.shape[0]):
            for j in range(offspring_brain.weights_input_hidden.shape[1]):
                if random.random() < 0.5:
                    offspring_brain.weights_input_hidden[i, j] = parent1.brain.weights_input_hidden[i, j]
                else:
                    offspring_brain.weights_input_hidden[i, j] = parent2.brain.weights_input_hidden[i, j]
        
        # Hidden-to-output weights crossover
        for i in range(offspring_brain.weights_hidden_output.shape[0]):
            for j in range(offspring_brain.weights_hidden_output.shape[1]):
                if random.random() < 0.5:
                    offspring_brain.weights_hidden_output[i, j] = parent1.brain.weights_hidden_output[i, j]
                else:
                    offspring_brain.weights_hidden_output[i, j] = parent2.brain.weights_hidden_output[i, j]
        
        # Bias crossover
        for i in range(len(offspring_brain.bias_hidden)):
            if random.random() < 0.5:
                offspring_brain.bias_hidden[i] = parent1.brain.bias_hidden[i]
            else:
                offspring_brain.bias_hidden[i] = parent2.brain.bias_hidden[i]
        
        for i in range(len(offspring_brain.bias_output)):
            if random.random() < 0.5:
                offspring_brain.bias_output[i] = parent1.brain.bias_output[i]
            else:
                offspring_brain.bias_output[i] = parent2.brain.bias_output[i]
        
        # Apply mutation to offspring
        if random.random() < self.config.mutation_rate:
            offspring_brain.mutate()
            
            # Track additional mutation event
            self.genetic_events['mutations'].append({
                'timestamp': time.time(),
                'type': 'post_crossover_mutation',
                'parent_species': f"{parent1.species_type.value}+{parent2.species_type.value}",
                'parent_fitness': (parent1.brain.fitness_score + parent2.brain.fitness_score) / 2,
                'generation': self.generation
            })
            self.genetic_events['recent_mutations'] += 1
            self.genetic_events['total_mutations'] += 1
        
        return offspring_brain
    
    def create_next_generation(self, current_agents: List[NeuralAgent]) -> List[NeuralAgent]:
        """Create next generation using genetic algorithm"""
        next_generation = []
        
        # Process each species separately
        for species_type in [SpeciesType.HERBIVORE, SpeciesType.CARNIVORE]:
            species_agents = [a for a in current_agents 
                            if a.species_type == species_type and a.is_alive]
            
            if not species_agents:
                continue
            
            # Determine target population for this species
            species_count = len([a for a in current_agents if a.species_type == species_type])
            target_count = max(10, species_count)  # Minimum viable population
            
            # Apply elitism - best agents automatically survive
            elite_agents = self.apply_elitism(species_agents, species_type)
            
            # Create copies of elite agents for next generation
            for elite in elite_agents:
                elite_copy = NeuralAgent(species_type, 
                                       Position(elite.position.x, elite.position.y), 
                                       random.randint(10000, 99999))
                elite_copy.brain = elite.brain.copy()
                # Reset tracking variables for new generation
                elite_copy.lifetime_energy_gained = 0
                elite_copy.lifetime_food_consumed = 0
                elite_copy.lifetime_successful_hunts = 0
                elite_copy.offspring_count = 0
                elite_copy.survival_time = 0
                next_generation.append(elite_copy)
            
            # Fill remaining population through breeding
            while len([a for a in next_generation if a.species_type == species_type]) < target_count:
                # Select parents through tournament selection
                parent1 = self.selection_tournament(species_agents, species_type)
                parent2 = self.selection_tournament(species_agents, species_type)
                
                if parent1 and parent2:
                    # Create offspring
                    offspring_pos = Position(
                        random.uniform(20, 180),
                        random.uniform(20, 180)
                    )
                    offspring = NeuralAgent(species_type, offspring_pos, random.randint(10000, 99999))
                    
                    # Apply genetic operators
                    offspring.brain = self.crossover_networks(parent1, parent2)
                    
                    next_generation.append(offspring)
        
        return next_generation
    
    def get_genetic_stats(self) -> Dict:
        """Get current genetic operation statistics"""
        return {
            'recent_mutations': self.genetic_events['recent_mutations'],
            'recent_crossovers': self.genetic_events['recent_crossovers'],
            'total_mutations': self.genetic_events['total_mutations'],
            'total_crossovers': self.genetic_events['total_crossovers'],
            'mutation_rate': len([e for e in self.genetic_events['mutations'] 
                                 if time.time() - e['timestamp'] < 60]),  # Last minute
            'crossover_rate': len([e for e in self.genetic_events['crossovers'] 
                                  if time.time() - e['timestamp'] < 60]),  # Last minute
            'recent_events': (self.genetic_events['mutations'][-10:] + 
                            self.genetic_events['crossovers'][-10:])
        }
    
    def reset_recent_counters(self):
        """Reset recent event counters for next update cycle"""
        self.genetic_events['recent_mutations'] = 0
        self.genetic_events['recent_crossovers'] = 0
        
        # Keep only recent events (last 5 minutes) to prevent memory buildup
        # NOTE: We do NOT update total_mutations/total_crossovers here because
        # those should represent the true total since simulation start
        current_time = time.time()
        self.genetic_events['mutations'] = [
            e for e in self.genetic_events['mutations'] 
            if current_time - e['timestamp'] < 300
        ]
        self.genetic_events['crossovers'] = [
            e for e in self.genetic_events['crossovers'] 
            if current_time - e['timestamp'] < 300
        ]
    
    def run_generation(self, environment: NeuralEnvironment) -> Dict:
        """Run a single generation of evolution"""
        print(f"\n🧬 Generation {self.generation + 1}")
        print("=" * 50)
        
        generation_stats = {
            "generation": self.generation,
            "initial_population": len(environment.agents),
            "steps_survived": 0
        }
        
        # Run simulation for specified number of steps
        for step in range(self.config.generation_length):
            environment.update()
            
            # Update fitness for all living agents
            for agent in environment.agents:
                if agent.is_alive:
                    agent.update_fitness(environment)
                    # Add diversity bonus to fitness
                    diversity_bonus = self.calculate_diversity_bonus(agent, environment.agents)
                    agent.brain.fitness_score += diversity_bonus
            
            # Print progress periodically
            if step % 200 == 0:
                alive_count = sum(1 for a in environment.agents if a.is_alive)
                print(f"⏱️ Step {step}: {alive_count} agents alive")
        
        # Evaluate final population
        fitness_stats = self.evaluate_population(environment.agents)
        generation_stats.update(fitness_stats)
        
        # Print generation summary
        print("\n📊 Generation Summary:")
        for species in ["herbivore", "carnivore"]:
            if fitness_stats[species]["fitness"]:
                print(f"🌿 {species.capitalize()}s - "
                      f"Count: {len(fitness_stats[species]['fitness'])}, "
                      f"Avg Fitness: {fitness_stats[species]['mean']:.1f}, "
                      f"Best: {fitness_stats[species]['max']:.1f}")
        
        # Store generation history
        self.evolution_history.append(generation_stats)
        
        # Create next generation
        if self.generation < self.config.max_generations - 1:
            next_gen_agents = self.create_next_generation(environment.agents)
            environment.agents = next_gen_agents
            print(f"🆕 Next generation created: {len(next_gen_agents)} agents")
        
        self.generation += 1
        return generation_stats

class EvolutionaryEnvironment(NeuralEnvironment):
    """Environment specifically designed for evolutionary simulations"""
    
    def __init__(self, evolution_config: EvolutionConfig, width: int = 200, height: int = 200):
        super().__init__(width, height, use_neural_agents=False)
        
        self.evolution_config = evolution_config
        self.genetic_algorithm = GeneticAlgorithm(evolution_config)
        
        # Create initial population with genetic diversity
        self._create_diverse_initial_population()
        
        # Enhanced food system for longer simulations
        self.max_food_count = 60
        self._ensure_adequate_food()
    
    def _create_diverse_initial_population(self):
        """Create genetically diverse initial population"""
        self.agents = []
        
        # Create diverse neural network configurations
        base_config = NeuralNetworkConfig(
            input_size=8,
            hidden_size=12,
            output_size=4,
            mutation_rate=self.evolution_config.mutation_rate,
            mutation_strength=self.evolution_config.mutation_strength
        )
        
        # Create herbivores with diverse starting genetics
        herbivore_count = int(self.evolution_config.population_size * 0.6)  # 60% herbivores
        for i in range(herbivore_count):
            pos = Position(
                random.uniform(20, self.width - 20),
                random.uniform(20, self.height - 20)
            )
            agent = NeuralAgent(SpeciesType.HERBIVORE, pos, self.next_agent_id, base_config)
            
            # Add initial genetic diversity
            for _ in range(random.randint(0, 3)):
                agent.brain.mutate()
            
            self.agents.append(agent)
        
        # Create carnivores with diverse starting genetics
        carnivore_count = self.evolution_config.population_size - herbivore_count
        for i in range(carnivore_count):
            pos = Position(
                random.uniform(20, self.width - 20),
                random.uniform(20, self.height - 20)
            )
            agent = NeuralAgent(SpeciesType.CARNIVORE, pos, self.next_agent_id, base_config)
            
            # Add initial genetic diversity
            for _ in range(random.randint(0, 3)):
                agent.brain.mutate()
            
            self.agents.append(agent)
    
    def _ensure_adequate_food(self):
        """Ensure adequate food for larger populations"""
        while len(self.food_sources) < self.max_food_count:
            self._add_food_source()
    
    def _add_food_source(self):
        """Add a new food source to the environment"""
        from src.core.ecosystem import Food, Position
        pos = Position(
            random.uniform(10, self.width - 10),
            random.uniform(10, self.height - 10)
        )
        self.food_sources.append(Food(pos))
    
    def update(self):
        """Enhanced update with better food management"""
        self.step()  # Call the parent's step method
        
        # Maintain food levels for longer simulations
        if len(self.food_sources) < self.max_food_count // 2:
            for _ in range(5):  # Add multiple food sources at once
                self._add_food_source()

def run_evolutionary_simulation():
    """Run complete evolutionary simulation"""
    print("🧬 Starting Phase 3: Evolution & Genetic Algorithms")
    print("=" * 60)
    
    # Configure evolution parameters
    evolution_config = EvolutionConfig(
        population_size=40,
        elite_percentage=0.15,  # Top 15% survive
        mutation_rate=0.25,
        mutation_strength=0.4,
        crossover_rate=0.8,
        tournament_size=4,
        generation_length=800,
        max_generations=20,
        fitness_sharing=True,
        diversity_bonus=0.2
    )
    
    # Create evolutionary environment
    env = EvolutionaryEnvironment(evolution_config)
    
    print(f"🌱 Initial population: {len(env.agents)} agents")
    print(f"🎯 Evolution parameters:")
    print(f"   • Generations: {evolution_config.max_generations}")
    print(f"   • Steps per generation: {evolution_config.generation_length}")
    print(f"   • Elite survival rate: {evolution_config.elite_percentage*100:.0f}%")
    print(f"   • Mutation rate: {evolution_config.mutation_rate*100:.0f}%")
    print(f"   • Crossover rate: {evolution_config.crossover_rate*100:.0f}%")
    
    # Run evolution
    try:
        for generation in range(evolution_config.max_generations):
            generation_stats = env.genetic_algorithm.run_generation(env)
            
            # Check for extinction
            alive_agents = [a for a in env.agents if a.is_alive]
            if len(alive_agents) < 5:
                print("⚠️ Population too small, ending evolution early")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Evolution interrupted by user")
    
    # Print final results
    print("\n" + "=" * 60)
    print("🏆 EVOLUTION COMPLETE - Final Results")
    print("=" * 60)
    
    # Print hall of fame
    for species in ["herbivore", "carnivore"]:
        hall = env.genetic_algorithm.hall_of_fame[species]
        if hall:
            best = max(hall, key=lambda x: x["fitness"])
            print(f"🥇 Best {species}: Fitness {best['fitness']:.1f} "
                  f"(Gen {best['generation']}, Survived {best['survival_time']} steps)")
    
    # Print evolution history summary
    if env.genetic_algorithm.evolution_history:
        print(f"\n📈 Evolution Summary:")
        print(f"   • Generations completed: {len(env.genetic_algorithm.evolution_history)}")
        
        # Calculate fitness trends
        herbivore_fitness = [gen.get("herbivore", {}).get("mean", 0) 
                           for gen in env.genetic_algorithm.evolution_history]
        carnivore_fitness = [gen.get("carnivore", {}).get("mean", 0) 
                           for gen in env.genetic_algorithm.evolution_history]
        
        if herbivore_fitness and any(f > 0 for f in herbivore_fitness):
            print(f"   • Herbivore fitness: {herbivore_fitness[0]:.1f} → {herbivore_fitness[-1]:.1f}")
        if carnivore_fitness and any(f > 0 for f in carnivore_fitness):
            print(f"   • Carnivore fitness: {carnivore_fitness[0]:.1f} → {carnivore_fitness[-1]:.1f}")
    
    return env.genetic_algorithm.evolution_history, env.genetic_algorithm.hall_of_fame

if __name__ == "__main__":
    # Test evolutionary system
    evolution_history, hall_of_fame = run_evolutionary_simulation()
    
    print("\n🚀 Phase 3 Evolution system ready!")
    print("Use run_evolutionary_simulation() to start full evolution")
