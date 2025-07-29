"""
Enhanced Agent class with Neural Network decision-making
Extends the original ecosystem with AI-driven behaviors
"""
from src.core.ecosystem import Agent, SpeciesType, Position, Environment
from src.neural.neural_network import NeuralNetwork, NeuralNetworkConfig, SensorSystem
import random
import math
import time
import numpy as np

class NeuralAgent(Agent):
    """Agent with neural network-based decision making"""
    
    def __init__(self, species_type: SpeciesType, position: Position, agent_id: int, 
                 neural_config: NeuralNetworkConfig = None, generation: int = 1):
        super().__init__(species_type, position, agent_id)
        
        # Initialize neural network
        if neural_config is None:
            neural_config = NeuralNetworkConfig()
        
        self.brain = NeuralNetwork(neural_config)
        
        # Generation tracking
        self.generation = generation  # Which generation this agent belongs to
        
        # Performance tracking for evolution
        self.lifetime_energy_gained = 0
        self.lifetime_food_consumed = 0
        self.lifetime_successful_hunts = 0
        self.lifetime_escapes = 0
        self.offspring_count = 0
        self.survival_time = 0
        
        # Behavioral modifiers
        self.last_action = None
        self.action_cooldown = 0
        self.was_hunted = False  # Track if hunted this step
        
    def make_neural_decision(self, environment: Environment) -> dict:
        """Use neural network to make decisions"""
        # Get sensory inputs
        sensory_inputs = SensorSystem.get_sensory_inputs(self, environment)
        
        # Process through neural network
        neural_outputs = self.brain.forward(np.array(sensory_inputs))
        
        # Interpret outputs as actions
        actions = SensorSystem.interpret_network_output(neural_outputs)
        
        return actions
    
    def neural_move(self, environment: Environment):
        """Execute movement based on neural network decision"""
        try:
            actions = self.make_neural_decision(environment)
            
            # Extract movement components
            move_x = actions['move_x']
            move_y = actions['move_y']
            intensity = actions['intensity']
            should_reproduce = actions['should_reproduce']
            
            # Apply movement with species-specific speed
            move_distance = self.speed * intensity * 0.5  # Scale down for stability
            
            # Normalize movement vector
            movement_magnitude = math.sqrt(move_x**2 + move_y**2)
            if movement_magnitude > 0:
                move_x /= movement_magnitude
                move_y /= movement_magnitude
            
            # Apply movement
            self.position.x += move_x * move_distance
            self.position.y += move_y * move_distance
            
            # Handle reproduction decision
            if should_reproduce and self.can_reproduce() and self.action_cooldown <= 0:
                return self.neural_reproduce()
            
        except Exception as e:
            # Fallback to random movement if neural network fails
            self.random_move()
            return None
        
        return None
    
    def neural_reproduce(self) -> 'NeuralAgent':
        """Create offspring with inherited and mutated neural network"""
        if not self.can_reproduce():
            return None
        
        # Use parent's reproduction mechanics
        self.energy -= self.reproduction_cost
        self.reproduction_cooldown = 200
        self.action_cooldown = 50  # Prevent rapid decisions
        
        # Create offspring position
        offspring_pos = Position(
            self.position.x + random.uniform(-8, 8),
            self.position.y + random.uniform(-8, 8)
        )
        
        # Create offspring with inherited brain and next generation
        offspring = NeuralAgent(
            self.species_type, 
            offspring_pos, 
            random.randint(10000, 99999),
            generation=self.generation + 1  # Increment generation
        )
        
        # Inherit parent's neural network with mutations
        offspring.brain = self.brain.copy()
        offspring.brain.mutate()
        
        # Track reproduction success
        self.offspring_count += 1
        
        return offspring
    
    def update_fitness(self, environment: Environment):
        """Update neural network fitness based on agent performance"""
        self.survival_time += 1
        
        # Base fitness from survival
        survival_fitness = self.survival_time * 0.1
        
        # Energy management fitness
        energy_fitness = (self.energy / self.max_energy) * 10
        
        # Species-specific bonuses
        species_bonus = 0
        if self.species_type == SpeciesType.HERBIVORE:
            # Bonus for finding food efficiently
            species_bonus = self.lifetime_food_consumed * 2
            # Penalty for being caught (low energy from threats)
            if hasattr(self, 'was_hunted') and self.was_hunted:
                species_bonus -= 20
        else:  # Carnivore
            # Bonus for successful hunts
            species_bonus = self.lifetime_successful_hunts * 15
            # Bonus for maintaining energy through hunting
            if self.energy > self.max_energy * 0.7:
                species_bonus += 5
        
        # Reproduction fitness
        reproduction_fitness = self.offspring_count * 25
        
        # Combine fitness components
        total_fitness = survival_fitness + energy_fitness + species_bonus + reproduction_fitness
        
        # Update neural network fitness (with momentum for stability)
        momentum = 0.9
        self.brain.fitness_score = (momentum * self.brain.fitness_score + 
                                   (1 - momentum) * total_fitness)
    
    def consume_food(self, food_energy: int):
        """Track food consumption for fitness - HERBIVORES ONLY"""
        # Species check: Only herbivores can consume plant food
        if self.species_type != SpeciesType.HERBIVORE:
            print(f"⚠️ WARNING: {self.species_type.value} (ID: {self.id}) attempted to consume plant food - blocked!")
            return
        
        self.energy = min(self.max_energy, self.energy + food_energy)
        self.lifetime_food_consumed += 1
        self.lifetime_energy_gained += food_energy
    
    def successful_hunt(self, energy_gained: int):
        """Track successful hunts for fitness"""
        self.energy = min(self.max_energy, self.energy + energy_gained)
        self.lifetime_successful_hunts += 1
        self.lifetime_energy_gained += energy_gained
    
    def update(self):
        """Enhanced update with neural network aging"""
        super().update()
        
        # Update action cooldown
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
        
        # Age the neural network
        self.brain.age += 1

class NeuralEnvironment(Environment):
    """Enhanced environment that works with neural agents"""
    
    def __init__(self, width: int = 200, height: int = 200, use_neural_agents: bool = True):
        # Initialize parent environment but don't create agents yet
        super().__init__(width, height)
        
        # Initialize genetic algorithm for tracking mutations/crossovers
        from src.evolution.genetic_evolution import GeneticAlgorithm, EvolutionConfig
        evolution_config = EvolutionConfig()
        self.genetic_algorithm = GeneticAlgorithm(evolution_config)
        
        # Clear the default agents and create neural agents instead
        if use_neural_agents:
            self.agents = []
            self._initialize_neural_agents()
            
        # Upgrade existing neural networks to support boundary awareness
        self._upgrade_neural_networks_for_boundary_awareness()
    
    def _upgrade_neural_networks_for_boundary_awareness(self):
        """Upgrade all neural agents to support boundary awareness"""
        for agent in self.agents:
            if isinstance(agent, NeuralAgent):
                agent.brain.upgrade_to_boundary_awareness()
    
    def _initialize_neural_agents(self):
        """Create initial population of neural agents"""
        neural_config = NeuralNetworkConfig(
            input_size=8,
            hidden_size=12,
            output_size=4,
            mutation_rate=0.15,
            mutation_strength=0.3
        )
        
        # Create herbivores with neural networks (Generation 1)
        for _ in range(20):
            pos = Position(
                random.uniform(20, self.width - 20),
                random.uniform(20, self.height - 20)
            )
            herbivore = NeuralAgent(SpeciesType.HERBIVORE, pos, self.next_agent_id, neural_config, generation=1)
            self.agents.append(herbivore)
            self.next_agent_id += 1
        
        # Create carnivores with neural networks (Generation 1) 
        for _ in range(8):
            pos = Position(
                random.uniform(20, self.width - 20),
                random.uniform(20, self.height - 20)
            )
            carnivore = NeuralAgent(SpeciesType.CARNIVORE, pos, self.next_agent_id, neural_config, generation=1)
            self.agents.append(carnivore)
            self.next_agent_id += 1
    
    def step(self):
        """Enhanced step function for neural agents"""
        self.time_step += 1
        new_agents = []
        
        # Update all neural agents
        for agent in self.agents:
            if not agent.is_alive:
                continue
            
            # Neural decision-making replaces rule-based behavior
            if isinstance(agent, NeuralAgent):
                # Reset hunting flag for this step
                agent.was_hunted = False
                
                # Use neural network for decision making
                offspring = agent.neural_move(self)
                if offspring:
                    new_agents.append(offspring)
                
                # Handle feeding behavior with neural decisions
                self._handle_neural_feeding(agent)
                
                # Update fitness for evolution
                agent.update_fitness(self)
            else:
                # Fallback to original behavior for non-neural agents
                self._handle_traditional_behavior(agent)
            
            # Update agent state
            agent.update()
            
            # Keep agent in bounds
            self.keep_agent_in_bounds(agent)
        
        # Add new offspring
        for new_agent in new_agents:
            self.keep_agent_in_bounds(new_agent)
            self.agents.append(new_agent)
            self.next_agent_id += 1
        
        # Remove dead agents
        self.agents = [agent for agent in self.agents if agent.is_alive]
        
        # Trigger evolutionary events during simulation
        self.trigger_evolutionary_events()
        
        # Regenerate food
        for food in self.food_sources:
            if not food.is_available:
                food.current_regen += 1
                if food.current_regen >= food.regeneration_time:
                    food.is_available = True
    
    def _handle_neural_feeding(self, agent: NeuralAgent):
        """Handle feeding behavior based on proximity and species"""
        if agent.species_type == SpeciesType.HERBIVORE:
            # Check for nearby food
            food = self.find_nearest_food(agent)
            if food and agent.position.distance_to(food.position) < 3.0:
                agent.consume_food(food.energy_value)
                food.is_available = False
                food.current_regen = 0
        
        else:  # Carnivore
            # Check for hunting opportunities
            prey = self.find_nearest_prey(agent)
            if prey and prey.is_alive and agent.position.distance_to(prey.position) < 4.0:
                # Ensure prey hasn't already been killed this step
                if getattr(prey, 'was_hunted', False):
                    return  # Prey already targeted this step
                
                # Neural-based hunting success
                hunt_success_chance = min(0.7, agent.energy / 120)
                if random.random() < hunt_success_chance:
                    # Mark prey as hunted immediately to prevent double-hunting
                    prey.was_hunted = True
                    
                    # Calculate energy gain from live prey
                    energy_gained = int(prey.energy * 0.8)  # Better energy transfer
                    
                    # Kill the prey first
                    prey.is_alive = False
                    
                    # Track carnivore consumption (hunting herbivores)
                    agent.lifetime_food_consumed += 1
                    
                    # Then give energy to hunter
                    agent.successful_hunt(energy_gained)
    
    def _handle_traditional_behavior(self, agent):
        """Fallback to original Agent behavior"""
        # This is the original behavior from ecosystem.py
        if agent.species_type == SpeciesType.HERBIVORE:
            threat = self.find_nearest_threat(agent)
            if threat and agent.position.distance_to(threat.position) < 15:
                agent.move_away_from(threat.position, speed_multiplier=1.5)
            else:
                food = self.find_nearest_food(agent)
                if food:
                    if agent.position.distance_to(food.position) < 2.0:
                        agent.consume_food(food.energy_value)  # Fixed: Use proper tracking method
                        food.is_available = False
                        food.current_regen = 0
                    else:
                        agent.move_towards(food.position)
                else:
                    agent.random_move()
        else:  # Carnivore
            prey = self.find_nearest_prey(agent)
            if prey:
                if self.attempt_hunt(agent, prey):
                    pass
                else:
                    agent.move_towards(prey.position, speed_multiplier=1.2)
            else:
                agent.random_move()
    
    def get_neural_stats(self) -> dict:
        """Get statistics specific to neural agents"""
        stats = self.get_population_stats()
        
        neural_agents = [agent for agent in self.agents if isinstance(agent, NeuralAgent)]
        
        if neural_agents:
            avg_fitness = sum(agent.brain.fitness_score for agent in neural_agents) / len(neural_agents)
            avg_decisions = sum(agent.brain.decisions_made for agent in neural_agents) / len(neural_agents)
            total_offspring = sum(agent.offspring_count for agent in neural_agents)
            
            stats.update({
                'neural_agents': len(neural_agents),
                'avg_neural_fitness': round(avg_fitness, 2),
                'avg_decisions_made': round(avg_decisions, 1),
                'total_offspring_produced': total_offspring
            })
            
        # Add genetic operation statistics
        genetic_stats = self.genetic_algorithm.get_genetic_stats()
        stats.update(genetic_stats)
        
        return stats
    
    def trigger_evolutionary_events(self):
        """Randomly trigger mutations and crossovers for demonstration"""
        neural_agents = [agent for agent in self.agents if isinstance(agent, NeuralAgent)]
        
        if len(neural_agents) < 2:
            return
        
        # Occasionally trigger random mutations (5% chance per step)
        if random.random() < 0.05:
            random_agent = random.choice(neural_agents)
            random_agent.brain.mutate()
            
            # Track the mutation
            self.genetic_algorithm.genetic_events['mutations'].append({
                'timestamp': time.time(),
                'type': 'random_mutation',
                'parent_species': random_agent.species_type.value,
                'parent_fitness': random_agent.brain.fitness_score,
                'generation': random_agent.generation  # Track parent's generation
            })
            self.genetic_algorithm.genetic_events['recent_mutations'] += 1
            self.genetic_algorithm.genetic_events['total_mutations'] += 1
        
        # Occasionally trigger crossovers when agents reproduce (3% chance)
        if random.random() < 0.03 and len(neural_agents) >= 2:
            parent1 = random.choice(neural_agents)
            parent2 = random.choice([a for a in neural_agents if a != parent1])
            
            if parent1.species_type == parent2.species_type and len(neural_agents) < 50:
                # Create offspring through crossover
                offspring_pos = Position(
                    random.uniform(20, self.width - 20),
                    random.uniform(20, self.height - 20)
                )
                # Use the highest generation from either parent + 1
                offspring_generation = max(parent1.generation, parent2.generation) + 1
                offspring = NeuralAgent(parent1.species_type, offspring_pos, self.next_agent_id, generation=offspring_generation)
                offspring.brain = self.genetic_algorithm.crossover_networks(parent1, parent2)
                
                self.agents.append(offspring)
                self.next_agent_id += 1
                
                # Update parent counts
                parent1.offspring_count += 1
                parent2.offspring_count += 1

if __name__ == "__main__":
    # Test neural environment
    print("🧠 Testing Neural Environment")
    print("=" * 50)
    
    env = NeuralEnvironment()
    print(f"✅ Neural environment created with {len(env.agents)} neural agents")
    
    # Run a few steps
    for step in range(10):
        env.step()
        stats = env.get_neural_stats()
        if step % 3 == 0:
            print(f"Step {step}: H={stats['herbivore_count']}, C={stats['carnivore_count']}, "
                  f"Fitness={stats.get('avg_neural_fitness', 0):.1f}")
    
    print("\n🚀 Neural agents working successfully!")
