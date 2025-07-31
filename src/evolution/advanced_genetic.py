"""
Advanced Genetic Algorithm with Adaptive Parameters and Speciation
Enhances evolution with more sophisticated genetic operations
"""
import numpy as np
import random
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from src.evolution.genetic_evolution import GeneticAlgorithm, EvolutionConfig
from src.neural.neural_agents import NeuralAgent
from src.neural.neural_network import NeuralNetwork
from src.core.ecosystem import SpeciesType

@dataclass
class AdvancedEvolutionConfig(EvolutionConfig):
    """Enhanced configuration for advanced genetic operations"""
    # Adaptive parameters
    adaptive_mutation: bool = True
    adaptive_crossover: bool = True
    fitness_scaling: bool = True
    
    # Advanced selection methods
    rank_selection: bool = False
    fitness_sharing: bool = True
    speciation_threshold: float = 3.0
    
    # Advanced genetic operators
    gaussian_mutation: bool = True
    uniform_crossover: bool = True
    multi_point_crossover: bool = True
    
    # Evolution pressure controls
    selection_pressure: float = 1.2
    diversity_pressure: float = 0.1
    complexity_penalty: float = 0.01
    
    # Novelty search
    novelty_search: bool = False
    novelty_weight: float = 0.3
    behavior_archive_size: int = 50

class AdvancedGeneticAlgorithm(GeneticAlgorithm):
    """Enhanced genetic algorithm with advanced operators and adaptive parameters"""
    
    def __init__(self, config: AdvancedEvolutionConfig):
        super().__init__(config)
        self.config = config
        
        # Adaptive parameter tracking
        self.performance_history = []
        self.diversity_history = []
        self.mutation_rates = {"herbivore": config.mutation_rate, "carnivore": config.mutation_rate}
        self.crossover_rates = {"herbivore": config.crossover_rate, "carnivore": config.crossover_rate}
        
        # Speciation tracking
        self.species = {"herbivore": [], "carnivore": []}
        
        # Novelty search components
        self.behavior_archive = []
        self.behavior_descriptors = []
    
    def calculate_adaptive_parameters(self, agents: List[NeuralAgent], species_type: str):
        """Dynamically adjust mutation and crossover rates based on population performance"""
        if not self.config.adaptive_mutation and not self.config.adaptive_crossover:
            return
        
        species_agents = [a for a in agents if a.species_type.value == species_type and a.is_alive]
        if len(species_agents) < 3:
            return
        
        # Calculate fitness statistics
        fitnesses = [a.brain.fitness_score for a in species_agents]
        fitness_std = np.std(fitnesses)
        fitness_mean = np.mean(fitnesses)
        
        # Calculate diversity (weight variance across population)
        weight_diversity = self._calculate_population_diversity(species_agents)
        
        # Adaptive mutation rate
        if self.config.adaptive_mutation:
            base_mutation = self.config.mutation_rate
            
            # Increase mutation if population is too similar (low diversity)
            if weight_diversity < 0.5:
                self.mutation_rates[species_type] = min(0.5, base_mutation * 1.5)
            # Decrease mutation if population is performing well and diverse
            elif fitness_std > 2.0 and weight_diversity > 1.0:
                self.mutation_rates[species_type] = max(0.05, base_mutation * 0.7)
            else:
                self.mutation_rates[species_type] = base_mutation
        
        # Adaptive crossover rate
        if self.config.adaptive_crossover:
            base_crossover = self.config.crossover_rate
            
            # Increase crossover when fitness is stagnating
            if len(self.performance_history) > 5:
                recent_improvement = fitness_mean - np.mean(self.performance_history[-5:])
                if recent_improvement < 0.5:  # Stagnation
                    self.crossover_rates[species_type] = min(0.9, base_crossover * 1.3)
                else:
                    self.crossover_rates[species_type] = base_crossover
    
    def _calculate_population_diversity(self, agents: List[NeuralAgent]) -> float:
        """Calculate genetic diversity of population"""
        if len(agents) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                # Calculate Euclidean distance between neural networks
                dist = self._calculate_neural_distance(agents[i].brain, agents[j].brain)
                total_distance += dist
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _calculate_neural_distance(self, brain1: NeuralNetwork, brain2: NeuralNetwork) -> float:
        """Calculate distance between two neural networks"""
        # Weight differences
        input_diff = np.sum(np.abs(brain1.weights_input_hidden - brain2.weights_input_hidden))
        output_diff = np.sum(np.abs(brain1.weights_hidden_output - brain2.weights_hidden_output))
        bias_diff = np.sum(np.abs(brain1.bias_hidden - brain2.bias_hidden))
        bias_diff += np.sum(np.abs(brain1.bias_output - brain2.bias_output))
        
        return input_diff + output_diff + bias_diff
    
    def advanced_selection(self, agents: List[NeuralAgent], species_type: SpeciesType, 
                          selection_count: int) -> List[NeuralAgent]:
        """Advanced selection with multiple methods"""
        species_agents = [a for a in agents if a.species_type == species_type and a.is_alive]
        
        if len(species_agents) <= selection_count:
            return species_agents
        
        if self.config.rank_selection:
            return self._rank_selection(species_agents, selection_count)
        elif self.config.fitness_sharing:
            return self._fitness_sharing_selection(species_agents, selection_count)
        else:
            return self._tournament_selection_enhanced(species_agents, selection_count)
    
    def _rank_selection(self, agents: List[NeuralAgent], count: int) -> List[NeuralAgent]:
        """Rank-based selection to maintain diversity"""
        # Sort by fitness
        sorted_agents = sorted(agents, key=lambda x: x.brain.fitness_score, reverse=True)
        
        # Assign selection probabilities based on rank
        ranks = np.arange(1, len(sorted_agents) + 1)
        selection_probs = ranks / np.sum(ranks)
        
        # Select agents based on rank probabilities
        selected_indices = np.random.choice(
            len(sorted_agents), 
            size=count, 
            replace=True, 
            p=selection_probs
        )
        
        return [sorted_agents[i] for i in selected_indices]
    
    def _fitness_sharing_selection(self, agents: List[NeuralAgent], count: int) -> List[NeuralAgent]:
        """Fitness sharing to promote diversity"""
        shared_fitness = []
        
        for agent in agents:
            # Calculate sharing denominator
            sharing_sum = 0.0
            for other in agents:
                distance = self._calculate_neural_distance(agent.brain, other.brain)
                if distance < self.config.speciation_threshold:
                    sharing_function = 1.0 - (distance / self.config.speciation_threshold)
                    sharing_sum += sharing_function
            
            # Calculate shared fitness
            shared_fit = agent.brain.fitness_score / max(1.0, sharing_sum)
            shared_fitness.append(shared_fit)
        
        # Select based on shared fitness
        total_shared_fitness = sum(shared_fitness)
        if total_shared_fitness == 0:
            return random.sample(agents, min(count, len(agents)))
        
        selection_probs = [f / total_shared_fitness for f in shared_fitness]
        selected_indices = np.random.choice(
            len(agents), 
            size=count, 
            replace=True, 
            p=selection_probs
        )
        
        return [agents[i] for i in selected_indices]
    
    def _tournament_selection_enhanced(self, agents: List[NeuralAgent], count: int) -> List[NeuralAgent]:
        """Enhanced tournament selection with diversity consideration"""
        selected = []
        
        for _ in range(count):
            # Select tournament participants
            tournament_size = min(self.config.tournament_size, len(agents))
            tournament = random.sample(agents, tournament_size)
            
            # Sort by fitness
            tournament.sort(key=lambda x: x.brain.fitness_score, reverse=True)
            
            # Select with diversity consideration
            if len(tournament) > 1 and random.random() < self.config.diversity_pressure:
                # Sometimes select second best for diversity
                selected.append(tournament[1])
            else:
                selected.append(tournament[0])
        
        return selected
    
    def advanced_crossover(self, parent1: NeuralAgent, parent2: NeuralAgent) -> NeuralNetwork:
        """Advanced crossover with multiple methods"""
        species_key = parent1.species_type.value
        crossover_rate = self.crossover_rates.get(species_key, self.config.crossover_rate)
        
        if random.random() > crossover_rate:
            # No crossover, return mutated copy
            offspring_brain = parent1.brain.copy()
            self._advanced_mutation(offspring_brain, species_key)
            return offspring_brain
        
        # Choose crossover method
        if self.config.uniform_crossover:
            return self._uniform_crossover(parent1, parent2, species_key)
        elif self.config.multi_point_crossover:
            return self._multi_point_crossover(parent1, parent2, species_key)
        else:
            return self._blend_crossover(parent1, parent2, species_key)
    
    def _uniform_crossover(self, parent1: NeuralAgent, parent2: NeuralAgent, species_key: str) -> NeuralNetwork:
        """Uniform crossover - each weight independently chosen from either parent"""
        config = parent1.brain.config
        offspring = NeuralNetwork(config)
        
        # Input-to-hidden weights
        mask = np.random.random(offspring.weights_input_hidden.shape) < 0.5
        offspring.weights_input_hidden = np.where(
            mask, parent1.brain.weights_input_hidden, parent2.brain.weights_input_hidden
        )
        
        # Hidden-to-output weights
        mask = np.random.random(offspring.weights_hidden_output.shape) < 0.5
        offspring.weights_hidden_output = np.where(
            mask, parent1.brain.weights_hidden_output, parent2.brain.weights_hidden_output
        )
        
        # Biases
        mask = np.random.random(offspring.bias_hidden.shape) < 0.5
        offspring.bias_hidden = np.where(
            mask, parent1.brain.bias_hidden, parent2.brain.bias_hidden
        )
        
        mask = np.random.random(offspring.bias_output.shape) < 0.5
        offspring.bias_output = np.where(
            mask, parent1.brain.bias_output, parent2.brain.bias_output
        )
        
        # Apply mutation
        self._advanced_mutation(offspring, species_key)
        
        return offspring
    
    def _multi_point_crossover(self, parent1: NeuralAgent, parent2: NeuralAgent, species_key: str) -> NeuralNetwork:
        """Multi-point crossover with random crossover points"""
        config = parent1.brain.config
        offspring = NeuralNetwork(config)
        
        # Flatten all weights for multi-point crossover
        parent1_weights = np.concatenate([
            parent1.brain.weights_input_hidden.flatten(),
            parent1.brain.weights_hidden_output.flatten(),
            parent1.brain.bias_hidden.flatten(),
            parent1.brain.bias_output.flatten()
        ])
        
        parent2_weights = np.concatenate([
            parent2.brain.weights_input_hidden.flatten(),
            parent2.brain.weights_hidden_output.flatten(),
            parent2.brain.bias_hidden.flatten(),
            parent2.brain.bias_output.flatten()
        ])
        
        # Create crossover points
        num_points = random.randint(1, 4)
        crossover_points = sorted(random.sample(range(len(parent1_weights)), num_points))
        
        # Perform crossover
        offspring_weights = parent1_weights.copy()
        use_parent2 = False
        
        for i, point in enumerate(crossover_points + [len(parent1_weights)]):
            start = crossover_points[i-1] if i > 0 else 0
            if use_parent2:
                offspring_weights[start:point] = parent2_weights[start:point]
            use_parent2 = not use_parent2
        
        # Reshape back to original network structure
        self._reshape_weights_to_network(offspring, offspring_weights)
        
        # Apply mutation
        self._advanced_mutation(offspring, species_key)
        
        return offspring
    
    def _blend_crossover(self, parent1: NeuralAgent, parent2: NeuralAgent, species_key: str) -> NeuralNetwork:
        """Blend crossover with random interpolation"""
        config = parent1.brain.config
        offspring = NeuralNetwork(config)
        
        # Random blend factor
        alpha = np.random.random()
        
        # Blend all weights
        offspring.weights_input_hidden = (
            alpha * parent1.brain.weights_input_hidden + 
            (1 - alpha) * parent2.brain.weights_input_hidden
        )
        
        offspring.weights_hidden_output = (
            alpha * parent1.brain.weights_hidden_output + 
            (1 - alpha) * parent2.brain.weights_hidden_output
        )
        
        offspring.bias_hidden = (
            alpha * parent1.brain.bias_hidden + 
            (1 - alpha) * parent2.brain.bias_hidden
        )
        
        offspring.bias_output = (
            alpha * parent1.brain.bias_output + 
            (1 - alpha) * parent2.brain.bias_output
        )
        
        # Apply mutation
        self._advanced_mutation(offspring, species_key)
        
        return offspring
    
    def _advanced_mutation(self, network: NeuralNetwork, species_key: str):
        """Advanced mutation with multiple strategies"""
        mutation_rate = self.mutation_rates.get(species_key, self.config.mutation_rate)
        
        if self.config.gaussian_mutation:
            self._gaussian_mutation(network, mutation_rate)
        else:
            # Use original mutation method
            network.mutate()
    
    def _gaussian_mutation(self, network: NeuralNetwork, mutation_rate: float):
        """Gaussian mutation with adaptive strength"""
        mutation_strength = self.config.mutation_strength
        
        # Adaptive mutation strength based on network performance
        if hasattr(network, 'fitness_score') and network.fitness_score > 20:
            mutation_strength *= 0.7  # Reduce mutation for good performers
        
        # Apply Gaussian mutations
        for weights in [network.weights_input_hidden, network.weights_hidden_output]:
            mask = np.random.random(weights.shape) < mutation_rate
            mutations = np.random.normal(0, mutation_strength, weights.shape)
            weights += mask * mutations
        
        for biases in [network.bias_hidden, network.bias_output]:
            mask = np.random.random(biases.shape) < mutation_rate
            mutations = np.random.normal(0, mutation_strength * 0.5, biases.shape)
            biases += mask * mutations
    
    def _reshape_weights_to_network(self, network: NeuralNetwork, flat_weights: np.ndarray):
        """Reshape flattened weights back to network structure"""
        idx = 0
        
        # Input-to-hidden weights
        size = network.weights_input_hidden.size
        network.weights_input_hidden = flat_weights[idx:idx+size].reshape(network.weights_input_hidden.shape)
        idx += size
        
        # Hidden-to-output weights
        size = network.weights_hidden_output.size
        network.weights_hidden_output = flat_weights[idx:idx+size].reshape(network.weights_hidden_output.shape)
        idx += size
        
        # Hidden biases
        size = network.bias_hidden.size
        network.bias_hidden = flat_weights[idx:idx+size].reshape(network.bias_hidden.shape)
        idx += size
        
        # Output biases
        size = network.bias_output.size
        network.bias_output = flat_weights[idx:idx+size].reshape(network.bias_output.shape)
    
    def evaluate_population_advanced(self, agents: List[NeuralAgent]) -> Dict:
        """Advanced population evaluation with novelty search"""
        # Basic evaluation
        fitness_stats = super().evaluate_population(agents)
        
        # Calculate novelty if enabled
        if self.config.novelty_search:
            self._calculate_novelty_scores(agents)
        
        # Update performance history
        for species in ["herbivore", "carnivore"]:
            if fitness_stats[species]["fitness"]:
                avg_fitness = fitness_stats[species]["mean"]
                self.performance_history.append(avg_fitness)
                
                # Keep history manageable
                if len(self.performance_history) > 100:
                    self.performance_history.pop(0)
        
        # Calculate and update adaptive parameters
        for species_type in ["herbivore", "carnivore"]:
            self.calculate_adaptive_parameters(agents, species_type)
        
        return fitness_stats
    
    def _calculate_novelty_scores(self, agents: List[NeuralAgent]):
        """Calculate novelty scores for behavioral diversity"""
        for agent in agents:
            if not agent.is_alive:
                continue
                
            # Extract behavior descriptor (simplified)
            behavior = self._extract_behavior_descriptor(agent)
            
            # Calculate novelty (distance to k-nearest neighbors)
            novelty = self._calculate_behavioral_novelty(behavior)
            
            # Combine with fitness
            original_fitness = agent.brain.fitness_score
            novel_fitness = ((1 - self.config.novelty_weight) * original_fitness + 
                           self.config.novelty_weight * novelty * 10)
            
            agent.brain.fitness_score = novel_fitness
    
    def _extract_behavior_descriptor(self, agent: NeuralAgent) -> np.ndarray:
        """Extract behavioral descriptor for novelty calculation"""
        # Simplified behavior descriptor based on agent's characteristics
        return np.array([
            agent.position.x / 100.0,  # Normalized position
            agent.position.y / 100.0,
            agent.energy / agent.max_energy,  # Energy efficiency
            agent.lifetime_food_consumed / max(1, agent.age),  # Food efficiency
            agent.offspring_count,  # Reproduction success
            agent.survival_time / max(1, agent.age)  # Survival efficiency
        ])
    
    def _calculate_behavioral_novelty(self, behavior: np.ndarray) -> float:
        """Calculate novelty based on behavioral descriptor"""
        if len(self.behavior_archive) < 5:
            # Not enough data for novelty calculation
            novelty = 1.0
        else:
            # Calculate distance to k-nearest neighbors
            k = min(5, len(self.behavior_archive))
            distances = []
            
            for archived_behavior in self.behavior_archive:
                distance = np.linalg.norm(behavior - archived_behavior)
                distances.append(distance)
            
            distances.sort()
            novelty = np.mean(distances[:k])  # Average distance to k-nearest
        
        # Add to archive if novel enough
        if novelty > 0.5:
            self.behavior_archive.append(behavior)
            
            # Keep archive size manageable
            if len(self.behavior_archive) > self.config.behavior_archive_size:
                self.behavior_archive.pop(0)
        
        return novelty
    
    def get_advanced_stats(self) -> Dict:
        """Get comprehensive statistics about the advanced genetic algorithm"""
        return {
            'adaptive_parameters': {
                'herbivore_mutation_rate': self.mutation_rates.get('herbivore', 0),
                'carnivore_mutation_rate': self.mutation_rates.get('carnivore', 0),
                'herbivore_crossover_rate': self.crossover_rates.get('herbivore', 0),
                'carnivore_crossover_rate': self.crossover_rates.get('carnivore', 0)
            },
            'performance_trend': {
                'recent_performance': self.performance_history[-10:] if self.performance_history else [],
                'performance_variance': np.var(self.performance_history) if self.performance_history else 0
            },
            'novelty_search': {
                'behavior_archive_size': len(self.behavior_archive),
                'novelty_enabled': self.config.novelty_search
            },
            'genetic_events': self.genetic_events
        }
