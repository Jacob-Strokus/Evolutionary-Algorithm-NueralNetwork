"""
Evolutionary Neural Agent
Agent designed for evolution-driven learning with enhanced capabilities
"""
import numpy as np
import math
import random
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

from src.core.ecosystem import SpeciesType, Position
from src.neural.evolutionary_network import EvolutionaryNeuralNetwork, EvolutionaryNetworkConfig
from src.neural.evolutionary_sensors import EvolutionarySensorSystem
from src.evolution.advanced_fitness import AdvancedFitnessEvaluator

@dataclass
class EvolutionaryAgentConfig:
    """Configuration for evolutionary agents"""
    max_energy: float = 100.0
    reproduction_cost: float = 50.0
    movement_cost_per_unit: float = 0.1
    age_energy_cost: float = 0.05  # Base energy cost for herbivores
    carnivore_energy_cost: float = 1.0  # Reduced energy cost for carnivores without food
    reproduction_cooldown: int = 20
    memory_tracking: bool = True
    social_learning: bool = True
    exploration_tracking: bool = True

class EvolutionaryNeuralAgent:
    """Enhanced neural agent with evolutionary learning capabilities"""
    
    def __init__(self, species_type: SpeciesType, position: Position, agent_id: int, 
                 config: Optional[EvolutionaryAgentConfig] = None,
                 network_config: Optional[EvolutionaryNetworkConfig] = None):
        
        self.species_type = species_type
        self.position = position
        self.agent_id = agent_id
        self.config = config or EvolutionaryAgentConfig()
        
        # Basic attributes
        self.energy = self.config.max_energy
        self.max_energy = self.config.max_energy
        self.age = 0
        self.is_alive = True
        self.reproduction_cooldown = 0
        
        # Species-specific attributes
        if species_type == SpeciesType.HERBIVORE:
            self.vision_range = 15.0
            self.speed = 1.5
        else:  # CARNIVORE
            self.vision_range = 30.0  # Increased vision for better prey tracking
            self.speed = 2.5  # Increased speed for better hunting
        
        # Enhanced neural network
        net_config = network_config or EvolutionaryNetworkConfig()
        self.brain = EvolutionaryNeuralNetwork(net_config)
        
        # Advanced fitness evaluator (can be overridden by environment's global evaluator)
        self.fitness_evaluator = None  # Will be set by environment
        
        # Memory and learning systems
        self.last_position = position
        self.visited_areas = defaultdict(int) if self.config.memory_tracking else None
        self.movement_history = deque(maxlen=10)
        self.food_history = deque(maxlen=5)
        self.social_encounters = deque(maxlen=20) if self.config.social_learning else None
        
        # Performance tracking
        self.total_distance_traveled = 0.0
        self.total_food_consumed = 0.0
        self.last_food_time = 0
        self.successful_reproductions = 0
        self.predator_escapes = 0
        self.prey_captures = 0
        
        # Communication system
        self.communication_output = 0.0
        self.received_signals = deque(maxlen=5)
        
        # Exploration system
        self.exploration_regions = set() if self.config.exploration_tracking else None
        self.novelty_bonus = 0.0
        
        # Starvation tracking for carnivores
        self.last_fed_time = 0  # Track when carnivore last ate
        self.steps_since_fed = 0
        
    def update(self, environment=None) -> Optional[Dict[str, Any]]:
        """Enhanced update with evolutionary learning"""
        if not self.is_alive:
            return None
        
        # If no environment provided, create a basic update
        if environment is None:
            # Basic agent update for compatibility with core ecosystem
            self.age += 1
            
            # Apply species-specific energy decay
            if self.species_type == SpeciesType.CARNIVORE:
                # Carnivores suffer more when they haven't eaten recently
                self.steps_since_fed += 1
                if self.steps_since_fed > 40:  # After 40 steps without eating, starvation begins
                    # Apply escalating starvation penalty
                    starvation_multiplier = 1.0 + (self.steps_since_fed - 40) * 0.06
                    energy_cost = self.config.carnivore_energy_cost * starvation_multiplier
                else:
                    energy_cost = self.config.age_energy_cost
                self.energy -= energy_cost
            else:
                # Herbivores have normal energy decay
                self.energy -= self.config.age_energy_cost
            
            if self.reproduction_cooldown > 0:
                self.reproduction_cooldown -= 1
            
            # Check if agent dies from energy depletion
            if self.energy <= 0:
                self.is_alive = False
            
            return None
        
        # Update position tracking
        self.last_position = Position(self.position.x, self.position.y)
        
        # Get enhanced sensory inputs
        sensory_inputs = EvolutionarySensorSystem.get_enhanced_sensory_inputs(self, environment)
        
        # Make neural decision with environmental context
        env_context = self._build_environment_context(environment)
        neural_outputs = self.brain.forward(np.array(sensory_inputs), env_context)
        actions = EvolutionarySensorSystem.interpret_enhanced_network_output(neural_outputs)
        
        # Execute actions
        action_results = self._execute_enhanced_actions(actions, environment)
        
        # Update memory and learning systems
        self._update_learning_systems(sensory_inputs, actions, action_results, environment)
        
        # Age and energy management
        self._age_and_energy_update()
        
        # Update fitness based on performance (with advanced evaluation)
        self._update_fitness_score(environment)
        
        return action_results
    
    def _build_environment_context(self, environment) -> Dict[str, Any]:
        """Build rich context information about environment state"""
        context = {
            'total_agents': len([a for a in environment.agents if a.is_alive]),
            'food_abundance': len(environment.food_sources),
            'predator_pressure': len([a for a in environment.agents 
                                    if a.is_alive and a.species_type == SpeciesType.CARNIVORE]),
            'prey_availability': len([a for a in environment.agents 
                                    if a.is_alive and a.species_type == SpeciesType.HERBIVORE]),
            'current_time': self.age
        }
        return context
    
    def _execute_enhanced_actions(self, actions: Dict[str, Any], environment) -> Dict[str, Any]:
        """Execute enhanced action set"""
        results = {
            'moved': False,
            'reproduced': False,
            'communicated': False,
            'food_consumed': 0.0,
            'distance_traveled': 0.0
        }
        
        # Movement with exploration bias
        move_x = actions['move_x']
        move_y = actions['move_y']
        intensity = actions['intensity']
        exploration_bias = actions['exploration_bias']
        
        # Apply exploration bias to movement
        if exploration_bias > 0.7 and self.brain.exploration_drive > 0.5:
            # Add random exploration component
            exploration_noise = np.random.normal(0, 0.3, 2)
            move_x += exploration_noise[0] * exploration_bias
            move_y += exploration_noise[1] * exploration_bias
            move_x = np.clip(move_x, -1, 1)
            move_y = np.clip(move_y, -1, 1)
        
        # Execute movement
        if abs(move_x) > 0.1 or abs(move_y) > 0.1:
            old_pos = Position(self.position.x, self.position.y)
            
            # Calculate new position
            movement_distance = self.speed * intensity * 0.5
            new_x = self.position.x + move_x * movement_distance
            new_y = self.position.y + move_y * movement_distance
            
            # Boundary handling (bounce back from walls)
            new_x = max(0, min(environment.width, new_x))
            new_y = max(0, min(environment.height, new_y))
            
            self.position = Position(new_x, new_y)
            
            # Track movement
            distance = old_pos.distance_to(self.position)
            self.total_distance_traveled += distance
            results['distance_traveled'] = distance
            results['moved'] = True
            
            # Energy cost for movement
            movement_cost = distance * self.config.movement_cost_per_unit
            self.energy -= movement_cost
            
            # Update visited areas
            if self.visited_areas is not None:
                area_key = (int(self.position.x // 10), int(self.position.y // 10))
                self.visited_areas[area_key] += 1
                
                # Exploration bonus for new areas
                if self.visited_areas[area_key] == 1:
                    self.novelty_bonus += 1.0
        
        # Communication
        self.communication_output = actions['communication_signal']
        if self.communication_output > 0.3:
            results['communicated'] = True
            # Broadcast signal to nearby agents
            self._broadcast_communication_signal(environment, self.communication_output)
        
        # Try to consume food
        food_consumed = self._attempt_food_consumption(environment)
        if food_consumed > 0:
            results['food_consumed'] = food_consumed
            self.total_food_consumed += food_consumed
            self.last_food_time = self.age
        
        # Reproduction
        if actions['reproduce'] and self.can_reproduce():
            reproduction_result = self._attempt_reproduction(environment)
            if reproduction_result:
                results['reproduced'] = True
                self.successful_reproductions += 1
        
        return results
    
    def _broadcast_communication_signal(self, environment, signal_strength: float):
        """Broadcast communication signal to nearby agents"""
        for other_agent in environment.agents:
            if (other_agent != self and other_agent.is_alive and 
                hasattr(other_agent, 'received_signals')):
                distance = self.position.distance_to(other_agent.position)
                if distance <= self.vision_range:
                    # Signal strength decreases with distance
                    received_strength = signal_strength * max(0.1, 1.0 - distance / self.vision_range)
                    other_agent.received_signals.append({
                        'strength': received_strength,
                        'source_species': self.species_type,
                        'source_id': self.agent_id,
                        'timestamp': self.age
                    })
    
    def _attempt_food_consumption(self, environment) -> float:
        """Attempt to consume nearby food"""
        if self.species_type == SpeciesType.HERBIVORE:
            return self._consume_plant_food(environment)
        else:
            return self._hunt_prey(environment)
    
    def _consume_plant_food(self, environment) -> float:
        """Consume plant food sources"""
        food_consumed = 0.0
        
        for food in environment.food_sources[:]:  # Copy list to avoid modification issues
            distance = self.position.distance_to(food.position)
            if distance <= 2.0:  # Close enough to eat
                # Energy gain with diminishing returns
                energy_gain = min(20.0, food.energy_value)
                self.energy = min(self.max_energy, self.energy + energy_gain)
                food_consumed += energy_gain
                
                # Remove consumed food
                environment.food_sources.remove(food)
                
                # Stop movement briefly when consuming food (80% reduction)
                if hasattr(self, 'movement_reduction'):
                    self.movement_reduction = 0.8
                
                break  # One food source per turn
        
        return food_consumed
    
    def _hunt_prey(self, environment) -> float:
        """Hunt herbivore prey with improved mechanics"""
        food_consumed = 0.0
        max_hunts_per_turn = 2  # Allow multiple hunts if very close
        hunts_this_turn = 0
        
        # Sort prey by distance to prioritize closest targets
        prey_list = []
        for prey in environment.agents:
            if (prey != self and prey.is_alive and 
                prey.species_type == SpeciesType.HERBIVORE):
                distance = self.position.distance_to(prey.position)
                if distance <= 7.0:  # Much larger hunting range
                    prey_list.append((distance, prey))
        
        # Sort by distance (closest first)
        prey_list.sort(key=lambda x: x[0])
        
        for distance, prey in prey_list:
            if hunts_this_turn >= max_hunts_per_turn:
                break
                
            # Calculate hunt success probability
            energy_factor = min(0.9, self.energy / 60)  # Better when well-fed
            distance_factor = max(0.3, 1.0 - (distance / 7.0))  # Closer = better
            health_factor = 0.3 if prey.energy < 50 else 0.1  # Easier to hunt weak prey
            
            # Base success rate around 85% for healthy carnivores
            hunt_success_chance = min(0.95, 0.75 + energy_factor * 0.15 + distance_factor * 0.15 + health_factor)
            
            if random.random() < hunt_success_chance:
                # Successful hunt
                energy_gain = min(50.0, prey.energy * 0.85)  # Better energy transfer
                self.energy = min(self.max_energy, self.energy + energy_gain)
                food_consumed += energy_gain
                
                # Reset starvation tracking
                self.steps_since_fed = 0
                self.last_fed_time = environment.time_step if hasattr(environment, 'time_step') else 0
                
                # Kill prey
                prey.is_alive = False
                self.prey_captures += 1
                hunts_this_turn += 1
        
        return food_consumed
    
    def _attempt_reproduction(self, environment) -> bool:
        """Attempt to reproduce with enhanced mate selection"""
        if not self.can_reproduce():
            return False
        
        # Enhanced mate finding with neural network preferences
        mate = self._find_optimal_mate(environment)
        if mate:
            # Create offspring with evolutionary crossover
            offspring = self._create_enhanced_offspring(mate, environment)
            if offspring:
                environment.agents.append(offspring)
                
                # Energy cost
                self.energy -= self.config.reproduction_cost
                mate.energy -= self.config.reproduction_cost
                
                # Cooldown
                self.reproduction_cooldown = self.config.reproduction_cooldown
                mate.reproduction_cooldown = self.config.reproduction_cooldown
                
                return True
        
        return False
    
    def _find_optimal_mate(self, environment):
        """Find optimal mate using neural network compatibility"""
        potential_mates = []
        
        for other_agent in environment.agents:
            if (other_agent != self and other_agent.is_alive and
                other_agent.species_type == self.species_type and
                other_agent.can_reproduce()):
                
                distance = self.position.distance_to(other_agent.position)
                if distance <= self.vision_range:
                    # Evaluate mate quality based on fitness and neural compatibility
                    compatibility = self._evaluate_mate_compatibility(other_agent)
                    potential_mates.append((other_agent, compatibility, distance))
        
        if not potential_mates:
            return None
        
        # Select best mate based on compatibility and proximity
        potential_mates.sort(key=lambda x: x[1] - x[2] * 0.1, reverse=True)
        return potential_mates[0][0]
    
    def _evaluate_mate_compatibility(self, mate) -> float:
        """Evaluate compatibility with potential mate"""
        # Fitness-based compatibility
        fitness_compatibility = (self.brain.fitness_score + mate.brain.fitness_score) / 2
        
        # Neural trait compatibility
        trait_compatibility = 1.0 - abs(self.brain.exploration_drive - mate.brain.exploration_drive)
        trait_compatibility += 1.0 - abs(self.brain.social_weight - mate.brain.social_weight)
        
        # Age compatibility (prefer similar ages)
        age_compatibility = 1.0 - abs(self.age - mate.age) / 1000.0
        
        return (fitness_compatibility + trait_compatibility + age_compatibility) / 3
    
    def _create_enhanced_offspring(self, mate, environment):
        """Create offspring through neural network crossover"""
        try:
            offspring_position = Position(
                (self.position.x + mate.position.x) / 2 + np.random.normal(0, 5),
                (self.position.y + mate.position.y) / 2 + np.random.normal(0, 5)
            )
            
            # Ensure offspring is within bounds
            offspring_position.x = max(0, min(environment.width, offspring_position.x))
            offspring_position.y = max(0, min(environment.height, offspring_position.y))
            
            # Create offspring agent
            offspring = EvolutionaryNeuralAgent(
                self.species_type,
                offspring_position,
                environment.get_next_agent_id(),
                self.config
            )
            
            # Neural network crossover
            offspring.brain = self.brain.crossover(mate.brain)
            
            # Apply mutations
            offspring.brain.mutate()
            
            # Inherit some traits with variation
            offspring.vision_range = (self.vision_range + mate.vision_range) / 2 + np.random.normal(0, 0.5)
            offspring.speed = (self.speed + mate.speed) / 2 + np.random.normal(0, 0.1)
            
            # Ensure reasonable ranges
            offspring.vision_range = max(10, min(25, offspring.vision_range))
            offspring.speed = max(0.5, min(3.0, offspring.speed))
            
            return offspring
            
        except Exception as e:
            print(f"Error creating offspring: {e}")
            return None
    
    def _update_learning_systems(self, sensory_inputs: List[float], actions: Dict[str, Any], 
                                results: Dict[str, Any], environment):
        """Update memory and learning systems"""
        # Store experience
        experience = {
            'inputs': sensory_inputs.copy(),
            'actions': actions.copy(),
            'results': results.copy(),
            'timestamp': self.age,
            'position': (self.position.x, self.position.y)
        }
        
        # Update movement history
        self.movement_history.append(experience)
        
        # Update food history
        if results['food_consumed'] > 0:
            self.food_history.append({
                'amount': results['food_consumed'],
                'position': (self.position.x, self.position.y),
                'timestamp': self.age
            })
        
        # Social learning
        if self.social_encounters is not None and self.received_signals:
            # Learn from successful agents' communication patterns
            for signal in self.received_signals:
                if signal['strength'] > 0.5:
                    # Strong signal suggests successful behavior
                    self.social_encounters.append(signal)
    
    def _age_and_energy_update(self):
        """Update age and apply aging effects"""
        self.age += 1
        
        # Age-related energy drain
        age_cost = self.config.age_energy_cost * (1 + self.age / 1000)
        self.energy -= age_cost
        
        # Reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        
        # Death from energy depletion
        if self.energy <= 0:
            self.is_alive = False
        
        # Death from old age (with some randomness)
        if self.age > 1500 and np.random.random() < 0.01:
            self.is_alive = False
    
    def _update_fitness_score(self, environment=None):
        """Update fitness score using advanced fitness evaluation"""
        if environment is not None and self.fitness_evaluator is not None:
            # Use advanced fitness evaluator
            fitness_components = self.fitness_evaluator.evaluate_agent_fitness(self, environment)
            
            # Store detailed fitness information for analysis
            if hasattr(self, 'detailed_fitness'):
                self.detailed_fitness = fitness_components
            else:
                self.detailed_fitness = fitness_components
            
            # Brain's fitness score is already updated by the evaluator
            return fitness_components
        else:
            # Fallback to basic fitness calculation for compatibility
            self._update_basic_fitness_score()
            return {"total_fitness": self.brain.fitness_score}
    
    def _update_basic_fitness_score(self):
        """Basic fitness calculation for backward compatibility"""
        # Base survival fitness
        survival_fitness = self.age / 10.0
        
        # Energy management fitness
        energy_fitness = self.energy / self.max_energy * 10.0
        
        # Food acquisition fitness
        food_fitness = self.total_food_consumed / max(1, self.age / 10) * 5.0
        
        # Reproduction fitness
        reproduction_fitness = self.successful_reproductions * 20.0
        
        # Movement efficiency fitness
        if self.total_distance_traveled > 0:
            efficiency_fitness = self.total_food_consumed / self.total_distance_traveled * 10.0
        else:
            efficiency_fitness = 0.0
        
        # Exploration bonus
        exploration_fitness = self.novelty_bonus * 2.0
        
        # Social interaction bonus
        social_fitness = len(self.received_signals) * 0.5 if self.received_signals else 0.0
        
        # Species-specific bonuses
        if self.species_type == SpeciesType.CARNIVORE:
            species_fitness = self.prey_captures * 15.0
        else:
            species_fitness = self.predator_escapes * 10.0
        
        # Combine all fitness components
        total_fitness = (survival_fitness + energy_fitness + food_fitness + 
                        reproduction_fitness + efficiency_fitness + 
                        exploration_fitness + social_fitness + species_fitness)
        
        self.brain.fitness_score = total_fitness
    
    def can_reproduce(self) -> bool:
        """Check if agent can reproduce"""
        base_requirements = (self.is_alive and 
                           self.energy >= self.config.reproduction_cost and
                           self.age >= 30 and  # Reduced age requirement
                           self.reproduction_cooldown <= 0)
        
        # Additional requirement for carnivores: must have eaten recently
        if self.species_type == SpeciesType.CARNIVORE:
            # Carnivores cannot reproduce if they haven't eaten in 60 steps
            return base_requirements and self.steps_since_fed < 60
        
        return base_requirements
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information"""
        brain_info = self.brain.get_network_info()
        
        return {
            'id': self.agent_id,
            'species': self.species_type.value,
            'position': (self.position.x, self.position.y),
            'energy': self.energy,
            'age': self.age,
            'is_alive': self.is_alive,
            'vision_range': self.vision_range,
            'speed': self.speed,
            'fitness_score': self.brain.fitness_score,
            'total_food_consumed': self.total_food_consumed,
            'total_distance_traveled': self.total_distance_traveled,
            'successful_reproductions': self.successful_reproductions,
            'neural_network': brain_info,
            'communication_output': self.communication_output,
            'novelty_bonus': self.novelty_bonus
        }

    def move_towards(self, target, speed_multiplier: float = 1.0):
        """Move towards a target position (required by core ecosystem)"""
        if self.position.distance_to(target) < 0.1:
            return
            
        dx = target.x - self.position.x
        dy = target.y - self.position.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            move_speed = self.speed * speed_multiplier
            self.position.x += (dx / distance) * move_speed
            self.position.y += (dy / distance) * move_speed
    
    def move_away_from(self, threat, speed_multiplier: float = 1.5):
        """Move away from a threatening position (required by core ecosystem)"""
        dx = self.position.x - threat.x
        dy = self.position.y - threat.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            move_speed = self.speed * speed_multiplier
            self.position.x += (dx / distance) * move_speed
            self.position.y += (dy / distance) * move_speed
    
    def random_move(self):
        """Move in a random direction"""
        import random
        angle = random.uniform(0, 2 * math.pi)
        move_speed = self.speed * 0.5
        self.position.x += math.cos(angle) * move_speed
        self.position.y += math.sin(angle) * move_speed

    def reproduce(self):
        """Create offspring with evolved neural network (required by core ecosystem)"""
        if not self.can_reproduce():
            return None
        
        # Deduct reproduction cost
        self.energy -= self.config.reproduction_cost
        self.reproduction_cooldown = self.config.reproduction_cooldown
        self.successful_reproductions += 1
        
        # Create offspring position nearby
        import random
        offspring_pos = Position(
            self.position.x + random.uniform(-5, 5),
            self.position.y + random.uniform(-5, 5)
        )
        
        # Create evolved offspring (simplified for ecosystem compatibility)
        offspring = EvolutionaryNeuralAgent(
            species_type=self.species_type,
            position=offspring_pos,
            agent_id=random.randint(10000, 99999),
            config=self.config
        )
        
        # Basic neural network inheritance (simplified)
        offspring.brain = self.brain.create_offspring()
        
        return offspring
