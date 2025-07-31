"""
AI Ecosystem Simulation - Phase 1
Basic environment with herbivores and carnivores
"""
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class SpeciesType(Enum):
    HERBIVORE = "herbivore"
    CARNIVORE = "carnivore"

@dataclass
class Position:
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Food:
    position: Position
    energy_value: int = 40  # Increased from 20 - more rewarding food!
    regeneration_time: int = 140  # Steps until food respawns
    current_regen: int = 0
    is_available: bool = True

class Agent:
    def __init__(self, species_type: SpeciesType, position: Position, agent_id: int):
        self.id = agent_id
        self.species_type = species_type
        self.position = position
        self.energy = 150  # Increased starting energy for better survival
        self.age = 0
        self.is_alive = True
        self.reproduction_cooldown = 0
        
        # Species-specific attributes (BALANCED FOR LEARNING)
        if species_type == SpeciesType.HERBIVORE:
            self.max_energy = 200  # Increased from 120
            self.reproduction_threshold = 120  # Increased from 80
            self.reproduction_cost = 50  # Increased from 40
            self.energy_decay = 0.5  # Reduced from 1 - KEY CHANGE!
            self.speed = 2.0
            self.vision_range = 15.0
            self.size = 1.0
        else:  # CARNIVORE
            self.max_energy = 250  # Increased from 150
            self.reproduction_threshold = 150  # Increased from 100
            self.reproduction_cost = 70  # Increased from 60
            self.energy_decay = 0.8  # Reduced from 2 - KEY CHANGE!
            self.speed = 3.0
            self.vision_range = 25.0  # Increased from 20.0 - Better hunting sight
            self.size = 1.5
    
    def move_towards(self, target: Position, speed_multiplier: float = 1.0):
        """Move towards a target position"""
        if self.position.distance_to(target) < 0.1:
            return
            
        dx = target.x - self.position.x
        dy = target.y - self.position.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            move_speed = self.speed * speed_multiplier
            self.position.x += (dx / distance) * move_speed
            self.position.y += (dy / distance) * move_speed
    
    def move_away_from(self, threat: Position, speed_multiplier: float = 1.5):
        """Move away from a threatening position"""
        dx = self.position.x - threat.x
        dy = self.position.y - threat.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            move_speed = self.speed * speed_multiplier
            self.position.x += (dx / distance) * move_speed
            self.position.y += (dy / distance) * move_speed
    
    def random_move(self):
        """Move in a random direction"""
        angle = random.uniform(0, 2 * math.pi)
        move_speed = self.speed * 0.5
        self.position.x += math.cos(angle) * move_speed
        self.position.y += math.sin(angle) * move_speed
    
    def can_reproduce(self) -> bool:
        return (self.energy >= self.reproduction_threshold and 
                self.reproduction_cooldown <= 0 and 
                self.age > 50)
    
    def reproduce(self) -> 'Agent':
        """Create a new agent of the same species"""
        if not self.can_reproduce():
            return None
            
        self.energy -= self.reproduction_cost
        self.reproduction_cooldown = 200
        
        # Offspring spawns nearby
        offspring_pos = Position(
            self.position.x + random.uniform(-5, 5),
            self.position.y + random.uniform(-5, 5)
        )
        
        return Agent(self.species_type, offspring_pos, random.randint(10000, 99999))
    
    def update(self):
        """Update agent state each time step"""
        self.age += 1
        self.energy -= self.energy_decay
        
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        
        # Death conditions
        if self.energy <= 0 or self.age > 2000:
            self.is_alive = False

class Environment:
    def __init__(self, width: int = 200, height: int = 200):
        self.width = width
        self.height = height
        self.agents: List[Agent] = []
        self.food_sources: List[Food] = []
        self.time_step = 0
        self.next_agent_id = 1
        
        self._initialize_food()
        self._initialize_agents()
    
    def _initialize_food(self):
        """Create initial food sources distributed across the environment"""
        num_food_sources = 50
        for _ in range(num_food_sources):
            pos = Position(
                random.uniform(10, self.width - 10),
                random.uniform(10, self.height - 10)
            )
            self.food_sources.append(Food(pos))
    
    def _initialize_agents(self):
        """Create initial population of herbivores and carnivores"""
        # Start with more herbivores than carnivores
        for _ in range(20):
            pos = Position(
                random.uniform(10, self.width - 10),
                random.uniform(10, self.height - 10)
            )
            herbivore = Agent(SpeciesType.HERBIVORE, pos, self.next_agent_id)
            self.agents.append(herbivore)
            self.next_agent_id += 1
        
        for _ in range(8):
            pos = Position(
                random.uniform(10, self.width - 10),
                random.uniform(10, self.height - 10)
            )
            carnivore = Agent(SpeciesType.CARNIVORE, pos, self.next_agent_id)
            self.agents.append(carnivore)
            self.next_agent_id += 1
    
    def add_agent(self, agent: Agent):
        """Add an agent to the environment"""
        self.agents.append(agent)
    
    def add_food(self, count: int = 1):
        """Add food sources to the environment with enhanced distribution"""
        # ENHANCED: Create clustered food distribution away from boundaries
        if count <= 8:  # For small amounts, use clustered approach
            self._add_clustered_food(count)
        else:  # For larger amounts, use mixed approach
            clusters = count // 4
            self._add_clustered_food(clusters * 4)
            # Add remaining scattered
            remaining = count - (clusters * 4)
            for _ in range(remaining):
                pos = Position(
                    random.uniform(self.width * 0.25, self.width * 0.75),
                    random.uniform(self.height * 0.25, self.height * 0.75)
                )
                food = Food(pos)
                food.regeneration_time = 150  # Slower regeneration
                self.food_sources.append(food)
    
    def _add_clustered_food(self, count: int):
        """Add food in clusters away from boundaries"""
        num_clusters = max(1, count // 4)
        items_per_cluster = count // num_clusters
        
        for cluster in range(num_clusters):
            # Place cluster center in safe zone (away from boundaries)
            center_x = random.uniform(self.width * 0.3, self.width * 0.7)
            center_y = random.uniform(self.height * 0.3, self.height * 0.7)
            
            # Create cluster of food around center
            for _ in range(items_per_cluster):
                # Cluster radius
                cluster_radius = 12
                offset_x = random.uniform(-cluster_radius, cluster_radius)
                offset_y = random.uniform(-cluster_radius, cluster_radius)
                
                # Ensure food stays in safe zone
                food_x = max(self.width * 0.2, min(self.width * 0.8, center_x + offset_x))
                food_y = max(self.height * 0.2, min(self.height * 0.8, center_y + offset_y))
                
                food = Food(Position(food_x, food_y))
                food.regeneration_time = 150  # Slower regeneration for balance
                self.food_sources.append(food)
    
    def get_neural_stats(self) -> dict:
        """Get neural network statistics for agents"""
        from src.neural.neural_agents import NeuralAgent
        
        neural_agents = [agent for agent in self.agents if isinstance(agent, NeuralAgent)]
        
        if not neural_agents:
            return {
                'herbivore_count': len([a for a in self.agents if a.species_type == SpeciesType.HERBIVORE]),
                'carnivore_count': len([a for a in self.agents if a.species_type == SpeciesType.CARNIVORE]),
                'avg_neural_fitness': 0,
                'avg_decisions_made': 0
            }
        
        herbivores = [a for a in neural_agents if a.species_type == SpeciesType.HERBIVORE]
        carnivores = [a for a in neural_agents if a.species_type == SpeciesType.CARNIVORE]
        
        total_fitness = sum(agent.brain.fitness_score for agent in neural_agents)
        avg_fitness = total_fitness / len(neural_agents) if neural_agents else 0
        
        total_decisions = sum(getattr(agent.brain, 'decisions_made', 0) for agent in neural_agents)
        avg_decisions = total_decisions / len(neural_agents) if neural_agents else 0
        
        return {
            'herbivore_count': len(herbivores),
            'carnivore_count': len(carnivores),
            'avg_neural_fitness': avg_fitness,
            'avg_decisions_made': avg_decisions,
            'total_agents': len(neural_agents)
        }
    
    def keep_agent_in_bounds(self, agent: Agent):
        """Ensure agent stays within environment boundaries"""
        agent.position.x = max(0, min(self.width, agent.position.x))
        agent.position.y = max(0, min(self.height, agent.position.y))
    
    def find_nearest_food(self, agent: Agent) -> Optional[Food]:
        """Find the nearest available food source within vision range"""
        nearest_food = None
        nearest_distance = float('inf')
        
        for food in self.food_sources:
            if food.is_available:
                distance = agent.position.distance_to(food.position)
                if distance <= agent.vision_range and distance < nearest_distance:
                    nearest_food = food
                    nearest_distance = distance
        
        return nearest_food
    
    def find_nearest_prey(self, predator: Agent) -> Optional[Agent]:
        """Find the nearest herbivore within vision range"""
        if predator.species_type != SpeciesType.CARNIVORE:
            return None
            
        nearest_prey = None
        nearest_distance = float('inf')
        
        for agent in self.agents:
            if (agent.species_type == SpeciesType.HERBIVORE and 
                agent.is_alive and agent != predator):
                distance = predator.position.distance_to(agent.position)
                if distance <= predator.vision_range and distance < nearest_distance:
                    nearest_prey = agent
                    nearest_distance = distance
        
        return nearest_prey
    
    def find_nearest_threat(self, prey: Agent) -> Optional[Agent]:
        """Find the nearest carnivore within vision range"""
        if prey.species_type != SpeciesType.HERBIVORE:
            return None
            
        nearest_threat = None
        nearest_distance = float('inf')
        
        for agent in self.agents:
            if (agent.species_type == SpeciesType.CARNIVORE and 
                agent.is_alive and agent != prey):
                distance = prey.position.distance_to(agent.position)
                if distance <= prey.vision_range and distance < nearest_distance:
                    nearest_threat = agent
                    nearest_distance = distance
        
        return nearest_threat
    
    def attempt_hunt(self, predator: Agent, prey: Agent) -> bool:
        """Attempt to hunt prey if close enough"""
        distance = predator.position.distance_to(prey.position)
        if distance <= 5.0:  # Increased hunt range
            # Improved success calculation based on multiple factors
            energy_factor = min(0.8, predator.energy / 80)  # Lower energy threshold
            size_factor = 0.9 if prey.energy < predator.energy else 0.6  # Easier to hunt smaller prey
            health_factor = 0.1 if prey.energy < 20 else 0.0  # Bonus for hunting weak prey
            
            hunt_success_chance = min(0.9, energy_factor + size_factor + health_factor)
            
            if random.random() < hunt_success_chance:
                # Successful hunt with better energy transfer
                energy_gained = int(prey.energy * 0.8)  # 80% energy transfer efficiency
                predator.energy = min(predator.max_energy, predator.energy + energy_gained)
                prey.is_alive = False
                return True
        return False
    
    def step(self):
        """Execute one simulation step"""
        self.time_step += 1
        new_agents = []
        
        # Update all agents
        for agent in self.agents:
            if not agent.is_alive:
                continue
                
            # Basic AI behavior
            if agent.species_type == SpeciesType.HERBIVORE:
                # Herbivore behavior: Find food, avoid predators, reproduce
                threat = self.find_nearest_threat(agent)
                if threat and agent.position.distance_to(threat.position) < 15:
                    # Run away from predators
                    agent.move_away_from(threat.position, speed_multiplier=1.5)
                else:
                    # Look for food
                    food = self.find_nearest_food(agent)
                    if food:
                        if agent.position.distance_to(food.position) < 2.0:
                            # Eat food
                            agent.energy = min(agent.max_energy, agent.energy + food.energy_value)
                            food.is_available = False
                            food.current_regen = 0
                        else:
                            # Move towards food
                            agent.move_towards(food.position)
                    else:
                        # Wander randomly
                        agent.random_move()
                
                # Try to reproduce
                if agent.can_reproduce():
                    offspring = agent.reproduce()
                    if offspring:
                        new_agents.append(offspring)
            
            else:  # CARNIVORE
                # Carnivore behavior: Hunt herbivores, reproduce
                prey = self.find_nearest_prey(agent)
                if prey:
                    # Try to hunt
                    if self.attempt_hunt(agent, prey):
                        pass  # Successful hunt handled in attempt_hunt
                    else:
                        # Move towards prey
                        agent.move_towards(prey.position, speed_multiplier=1.2)
                else:
                    # Wander randomly if no prey in sight
                    agent.random_move()
                
                # Try to reproduce
                if agent.can_reproduce():
                    offspring = agent.reproduce()
                    if offspring:
                        new_agents.append(offspring)
            
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
        
        # Regenerate food
        for food in self.food_sources:
            if not food.is_available:
                food.current_regen += 1
                if food.current_regen >= food.regeneration_time:
                    food.is_available = True
    
    def get_population_stats(self) -> dict:
        """Get current population statistics"""
        herbivore_count = sum(1 for agent in self.agents 
                            if agent.species_type == SpeciesType.HERBIVORE and agent.is_alive)
        carnivore_count = sum(1 for agent in self.agents 
                            if agent.species_type == SpeciesType.CARNIVORE and agent.is_alive)
        
        avg_herbivore_energy = 0
        avg_carnivore_energy = 0
        
        if herbivore_count > 0:
            avg_herbivore_energy = sum(agent.energy for agent in self.agents 
                                     if agent.species_type == SpeciesType.HERBIVORE and agent.is_alive) / herbivore_count
        
        if carnivore_count > 0:
            avg_carnivore_energy = sum(agent.energy for agent in self.agents 
                                     if agent.species_type == SpeciesType.CARNIVORE and agent.is_alive) / carnivore_count
        
        available_food = sum(1 for food in self.food_sources if food.is_available)
        
        return {
            'time_step': self.time_step,
            'herbivore_count': herbivore_count,
            'carnivore_count': carnivore_count,
            'total_population': herbivore_count + carnivore_count,
            'avg_herbivore_energy': round(avg_herbivore_energy, 1),
            'avg_carnivore_energy': round(avg_carnivore_energy, 1),
            'available_food': available_food
        }

if __name__ == "__main__":
    # Quick test when running ecosystem.py directly
    print("🧪 Testing Ecosystem Core Functionality")
    print("=" * 60)
    env = Environment()
    print(f"✅ Environment created: {len(env.agents)} agents, {len(env.food_sources)} food sources")
    
    # Run a quick simulation test
    print("🏃 Running 100 simulation steps...")
    for i in range(100):
        env.step()
        if i % 20 == 0:
            stats = env.get_population_stats()
            print(f"Step {i}: H={stats['herbivore_count']}, C={stats['carnivore_count']}")
    
    final_stats = env.get_population_stats()
    print(f"✅ Test complete! Final: H={final_stats['herbivore_count']}, C={final_stats['carnivore_count']}")
