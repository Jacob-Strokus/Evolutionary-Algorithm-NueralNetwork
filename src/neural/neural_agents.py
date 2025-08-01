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
        
        # Initialize neural network with food-seeking bias
        if neural_config is None:
            neural_config = NeuralNetworkConfig()
        
        self.brain = NeuralNetwork(neural_config)
        
        # BOUNDARY CLUSTERING FIX: Apply food-seeking bias to initial weights
        self._apply_food_seeking_bias()
        
        # Generation tracking
        self.generation = generation  # Which generation this agent belongs to
        
        # Performance tracking for evolution
        self.lifetime_energy_gained = 0
        self.lifetime_food_consumed = 0
        self.lifetime_successful_hunts = 0
        self.lifetime_escapes = 0
        self.offspring_count = 0
        self.survival_time = 0
        
        # Food acquisition tracking for fitness
        self.food_consumed_this_step = False
        self.last_food_distance = float('inf')
        
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
        
        # BOUNDARY CLUSTERING FIX: When standing on food (distance=0), reduce movement dramatically
        food_distance = sensory_inputs[2]  # Input [2] is food distance
        if food_distance <= 0.01:  # Standing on or very close to food
            # Reduce movement by 80% to encourage staying on food
            actions['move_x'] *= 0.2
            actions['move_y'] *= 0.2
            actions['intensity'] = max(0.1, actions['intensity'] * 0.3)  # Reduce movement intensity
        
        return actions
    
    def neural_move(self, environment: Environment):
        """Execute movement based on neural network decision"""
        try:
            actions = self.make_neural_decision(environment)
            
            # Track last action for fitness calculation
            self.last_action = actions
            
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
        """Update neural network fitness based on agent performance - FOCUSED ON FOOD ACQUISITION"""
        self.survival_time += 1
        
        # Base fitness from survival
        survival_fitness = self.survival_time * 0.1
        
        # Energy management fitness (still important for survival)
        energy_fitness = (self.energy / self.max_energy) * 8
        
        # FOOD-FOCUSED FITNESS: Core survival skill
        food_acquisition_fitness = self._calculate_food_acquisition_fitness(environment)
        
        # Species-specific bonuses focused on food efficiency
        species_bonus = 0
        if self.species_type == SpeciesType.HERBIVORE:
            # Strong bonus for food consumption efficiency
            species_bonus = self.lifetime_food_consumed * 5  # Increased from 2
            # Additional bonus for maintaining good energy through efficient foraging
            if self.lifetime_food_consumed > 0:
                food_efficiency = (self.energy / self.max_energy) / (self.survival_time / 100.0)
                species_bonus += food_efficiency * 15
            # Penalty for being caught (predation pressure)
            if hasattr(self, 'was_hunted') and self.was_hunted:
                species_bonus -= 20
        else:  # Carnivore
            # Strong bonus for successful hunts
            species_bonus = self.lifetime_successful_hunts * 25  # Increased from 20
            # Bonus for hunt efficiency (energy gained vs time spent)
            if self.lifetime_successful_hunts > 0:
                hunt_efficiency = (self.energy / self.max_energy) / (self.survival_time / 100.0)
                species_bonus += hunt_efficiency * 20
            # Base carnivore survival bonus
            species_bonus += 8
        
        # Reproduction fitness (natural selection reward)
        reproduction_fitness = self.offspring_count * 30  # Increased from 25
        
        # Combine fitness components (food acquisition is now primary focus)
        total_fitness = (survival_fitness + energy_fitness + species_bonus + 
                        reproduction_fitness + food_acquisition_fitness)
        
        # Update neural network fitness (with momentum for stability)
        momentum = 0.9
        self.brain.fitness_score = (momentum * self.brain.fitness_score + 
                                   (1 - momentum) * total_fitness)
    
    def _calculate_food_acquisition_fitness(self, environment: Environment):
        """Calculate fitness reward based on food acquisition abilities"""
        # Find nearest food source
        nearest_food_distance = float('inf')
        nearest_food = None
        
        for food in environment.food_sources:
            if food.is_available:
                distance = self.position.distance_to(food.position)
                if distance < nearest_food_distance:
                    nearest_food_distance = distance
                    nearest_food = food
        
        # Base food acquisition fitness
        food_fitness = 0
        
        # Reward for being close to food sources
        if nearest_food_distance < float('inf'):
            # Inverse distance reward - closer to food = higher fitness
            max_distance = math.sqrt(environment.width**2 + environment.height**2)
            proximity_ratio = 1.0 - (nearest_food_distance / max_distance)
            food_fitness += proximity_ratio * 15  # Strong reward for being near food
            
            # Extra bonus for being very close to food (within interaction range)
            if nearest_food_distance <= 5.0:  # Close enough to potentially consume
                food_fitness += 25  # Big bonus for being in food acquisition range
                
            # Bonus for moving towards food (check if getting closer)
            if hasattr(self, 'last_food_distance'):
                if nearest_food_distance < self.last_food_distance:
                    food_fitness += 10  # Reward for improving food approach
            self.last_food_distance = nearest_food_distance
        
        # Major bonus for actual food consumption
        if hasattr(self, 'food_consumed_this_step') and self.food_consumed_this_step:
            food_fitness += 50  # Huge reward for successful food acquisition
            self.food_consumed_this_step = False
        
        # BOUNDARY CLUSTERING FIX: Reward staying still when on food
        if nearest_food_distance < float('inf') and nearest_food_distance <= 2.0:
            # Agent is very close to or on food - reward low movement
            if hasattr(self, 'last_action') and self.last_action:
                movement_magnitude = (self.last_action.get('move_x', 0)**2 + 
                                    self.last_action.get('move_y', 0)**2)**0.5
                if movement_magnitude < 0.2:  # Very low movement when near food
                    food_fitness += 30  # Big reward for staying near food instead of wandering
        
        # For carnivores, consider prey proximity as "food"
        if self.species_type == SpeciesType.CARNIVORE:
            nearest_prey_distance = float('inf')
            for agent in environment.agents:
                if (agent.species_type == SpeciesType.HERBIVORE and 
                    agent.is_alive and agent != self):
                    distance = self.position.distance_to(agent.position)
                    if distance < nearest_prey_distance:
                        nearest_prey_distance = distance
            
            # Reward carnivores for being near potential prey
            if nearest_prey_distance < float('inf'):
                max_distance = math.sqrt(environment.width**2 + environment.height**2)
                prey_proximity_ratio = 1.0 - (nearest_prey_distance / max_distance)
                food_fitness += prey_proximity_ratio * 10  # Moderate reward for prey proximity
                
                # Extra bonus for being in hunting range
                if nearest_prey_distance <= 8.0:  # Within hunting range
                    food_fitness += 20
        
        return food_fitness
    
    def consume_food(self, food_energy: int):
        """Track food consumption for fitness - HERBIVORES ONLY"""
        # Species check: Only herbivores can consume plant food
        if self.species_type != SpeciesType.HERBIVORE:
            print(f"⚠️ WARNING: {self.species_type.value} (ID: {self.id}) attempted to consume plant food - blocked!")
            return
        
        self.energy = min(self.max_energy, self.energy + food_energy)
        self.lifetime_food_consumed += 1
        self.lifetime_energy_gained += food_energy
        
        # Mark food consumption for this step's fitness calculation
        self.food_consumed_this_step = True
    
    def successful_hunt(self, energy_gained: int):
        """Track successful hunts for fitness"""
        self.energy = min(self.max_energy, self.energy + energy_gained)
        self.lifetime_successful_hunts += 1
        self.lifetime_energy_gained += energy_gained
        
        # Mark food consumption (hunting) for this step's fitness calculation
        self.food_consumed_this_step = True
    
    def update(self):
        """Enhanced update with neural network aging"""
        super().update()
        
        # Update action cooldown
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
        
        # Age the neural network
        self.brain.age += 1
    
    def _apply_food_seeking_bias(self):
        """Apply initial weight bias to encourage food-seeking behavior over boundary clustering"""
        # Safety check: ensure network has the expected input size
        if self.brain.weights_input_hidden.shape[0] < 10:
            print(f"⚠️ Warning: Neural network has {self.brain.weights_input_hidden.shape[0]} inputs, expected 10. Skipping food-seeking bias.")
            return
        
        # Bias the input-to-hidden weights to prioritize food inputs
        food_distance_idx = 2  # Input [2] is food distance
        food_angle_idx = 3     # Input [3] is food angle  
        boundary_x_idx = 8     # Input [8] is X boundary distance
        boundary_y_idx = 9     # Input [9] is Y boundary distance
        
        # SOLUTION 1: Increase weights for food-related inputs (make food more important)
        self.brain.weights_input_hidden[food_distance_idx, :] *= 3.0  # Increased from 2.0
        self.brain.weights_input_hidden[food_angle_idx, :] *= 3.0     # Increased from 2.0
        
        # SOLUTION 2: Reduce weights for boundary inputs (prevent boundary obsession)
        self.brain.weights_input_hidden[boundary_x_idx, :] *= 0.2     # Reduced from 0.3
        self.brain.weights_input_hidden[boundary_y_idx, :] *= 0.2     # Reduced from 0.3     
        
        # SOLUTION 3: Initialize output biases to encourage food-seeking behavior
        # Small positive bias to encourage movement and exploration
        self.brain.bias_output[0] += 0.05  # Move X bias (slight exploration)
        self.brain.bias_output[1] += 0.05  # Move Y bias (slight exploration) 
        self.brain.bias_output[3] += 0.2   # Intensity bias (be more active)

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
            input_size=10,  # Updated for boundary awareness (was 8)
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
                # Reset step-based tracking flags
                agent.was_hunted = False
                agent.food_consumed_this_step = False
                
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
