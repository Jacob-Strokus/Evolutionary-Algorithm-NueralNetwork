"""
Advanced Fitness Systems for Phase 2
Dynamic fitness landscapes, competitive co-evolution, and adaptive optimization
"""

import numpy as np
import math
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from src.core.ecosystem import SpeciesType

class FitnessObjective(Enum):
    SURVIVAL = "survival"
    ENERGY_EFFICIENCY = "energy_efficiency"
    REPRODUCTION = "reproduction"
    EXPLORATION = "exploration"
    SOCIAL_COOPERATION = "social_cooperation"
    PREDATION = "predation"
    EVASION = "evasion"
    INNOVATION = "innovation"

@dataclass
class FitnessLandscape:
    """Dynamic fitness landscape configuration"""
    base_weights: Dict[FitnessObjective, float]
    environmental_modifiers: Dict[str, float]
    competition_pressure: float = 1.0
    resource_scarcity: float = 0.0
    predation_pressure: float = 0.0
    population_density: float = 0.0
    time_step: int = 0

@dataclass
class EcologicalNiche:
    """Represents an ecological niche specialization"""
    name: str
    fitness_emphasis: Dict[FitnessObjective, float]
    behavioral_traits: Dict[str, float]
    resource_preferences: List[str]
    optimal_population_size: int
    current_occupants: int = 0

class AdvancedFitnessEvaluator:
    """Advanced fitness evaluation with dynamic landscapes and co-evolution"""
    
    def __init__(self):
        # Define ecological niches
        self.niches = {
            "generalist": EcologicalNiche(
                name="Generalist",
                fitness_emphasis={
                    FitnessObjective.SURVIVAL: 0.3,
                    FitnessObjective.ENERGY_EFFICIENCY: 0.3,
                    FitnessObjective.REPRODUCTION: 0.2,
                    FitnessObjective.EXPLORATION: 0.2
                },
                behavioral_traits={"boldness": 0.5, "cooperation": 0.5, "innovation": 0.5},
                resource_preferences=["plants", "small_prey"],
                optimal_population_size=15
            ),
            "specialist_hunter": EcologicalNiche(
                name="Specialist Hunter",
                fitness_emphasis={
                    FitnessObjective.PREDATION: 0.4,
                    FitnessObjective.ENERGY_EFFICIENCY: 0.3,
                    FitnessObjective.SURVIVAL: 0.2,
                    FitnessObjective.REPRODUCTION: 0.1
                },
                behavioral_traits={"boldness": 0.8, "cooperation": 0.3, "innovation": 0.4},
                resource_preferences=["large_prey"],
                optimal_population_size=8
            ),
            "social_forager": EcologicalNiche(
                name="Social Forager",
                fitness_emphasis={
                    FitnessObjective.SOCIAL_COOPERATION: 0.4,
                    FitnessObjective.EXPLORATION: 0.3,
                    FitnessObjective.SURVIVAL: 0.2,
                    FitnessObjective.REPRODUCTION: 0.1
                },
                behavioral_traits={"boldness": 0.4, "cooperation": 0.9, "innovation": 0.6},
                resource_preferences=["distributed_resources"],
                optimal_population_size=20
            ),
            "innovator": EcologicalNiche(
                name="Innovator",
                fitness_emphasis={
                    FitnessObjective.INNOVATION: 0.4,
                    FitnessObjective.EXPLORATION: 0.3,
                    FitnessObjective.ENERGY_EFFICIENCY: 0.2,
                    FitnessObjective.SURVIVAL: 0.1
                },
                behavioral_traits={"boldness": 0.7, "cooperation": 0.4, "innovation": 0.9},
                resource_preferences=["novel_resources"],
                optimal_population_size=5
            )
        }
        
        # Current fitness landscape
        self.current_landscape = FitnessLandscape(
            base_weights={
                FitnessObjective.SURVIVAL: 1.0,
                FitnessObjective.ENERGY_EFFICIENCY: 0.8,
                FitnessObjective.REPRODUCTION: 0.6,
                FitnessObjective.EXPLORATION: 0.4,
                FitnessObjective.SOCIAL_COOPERATION: 0.3,
                FitnessObjective.PREDATION: 0.5,
                FitnessObjective.EVASION: 0.5,
                FitnessObjective.INNOVATION: 0.2
            },
            environmental_modifiers={}
        )
        
        # Tracking systems
        self.fitness_history = deque(maxlen=100)
        self.niche_assignments = {}  # agent_id -> niche_name
        self.competitive_rankings = defaultdict(list)
        self.innovation_tracking = defaultdict(int)
        
    def update_environmental_conditions(self, environment) -> None:
        """Update fitness landscape based on environmental conditions"""
        if not hasattr(environment, 'agents'):
            return
            
        # Count population by species
        herbivore_count = len([a for a in environment.agents 
                             if a.is_alive and a.species_type == SpeciesType.HERBIVORE])
        carnivore_count = len([a for a in environment.agents 
                             if a.is_alive and a.species_type == SpeciesType.CARNIVORE])
        total_agents = herbivore_count + carnivore_count
        
        # Calculate environmental pressures
        self.current_landscape.population_density = total_agents / max(1, environment.width * environment.height / 1000)
        self.current_landscape.predation_pressure = carnivore_count / max(1, herbivore_count) if herbivore_count > 0 else 0
        self.current_landscape.resource_scarcity = max(0, 1.0 - len(environment.food_sources) / max(1, total_agents))
        self.current_landscape.competition_pressure = min(2.0, total_agents / 20.0)
        self.current_landscape.time_step += 1
        
        # Update adaptive weights based on conditions
        self._update_adaptive_weights()
        
    def _update_adaptive_weights(self) -> None:
        """Dynamically adjust fitness weights based on environmental conditions"""
        landscape = self.current_landscape
        
        # High predation pressure emphasizes evasion and social cooperation
        if landscape.predation_pressure > 0.5:
            landscape.base_weights[FitnessObjective.EVASION] = min(1.5, 0.5 + landscape.predation_pressure)
            landscape.base_weights[FitnessObjective.SOCIAL_COOPERATION] = min(1.2, 0.3 + landscape.predation_pressure * 0.5)
        
        # Resource scarcity emphasizes efficiency and exploration
        if landscape.resource_scarcity > 0.3:
            landscape.base_weights[FitnessObjective.ENERGY_EFFICIENCY] = min(1.5, 0.8 + landscape.resource_scarcity)
            landscape.base_weights[FitnessObjective.EXPLORATION] = min(1.3, 0.4 + landscape.resource_scarcity * 0.8)
        
        # High population density emphasizes competition and innovation
        if landscape.population_density > 0.8:
            landscape.base_weights[FitnessObjective.INNOVATION] = min(1.0, 0.2 + landscape.population_density * 0.3)
            landscape.base_weights[FitnessObjective.REPRODUCTION] = max(0.3, 0.6 - landscape.population_density * 0.2)
        
        # Temporal dynamics - emphasize different traits over time
        time_cycle = (landscape.time_step % 500) / 500.0
        if time_cycle < 0.25:  # Early phase: exploration and growth
            landscape.base_weights[FitnessObjective.EXPLORATION] *= 1.3
            landscape.base_weights[FitnessObjective.REPRODUCTION] *= 1.2
        elif time_cycle < 0.5:  # Expansion phase: efficiency and competition
            landscape.base_weights[FitnessObjective.ENERGY_EFFICIENCY] *= 1.3
            landscape.base_weights[FitnessObjective.PREDATION] *= 1.2
        elif time_cycle < 0.75:  # Mature phase: cooperation and specialization
            landscape.base_weights[FitnessObjective.SOCIAL_COOPERATION] *= 1.4
            landscape.base_weights[FitnessObjective.INNOVATION] *= 1.3
        else:  # Reset phase: survival and adaptation
            landscape.base_weights[FitnessObjective.SURVIVAL] *= 1.2
            landscape.base_weights[FitnessObjective.EVASION] *= 1.1
    
    def evaluate_agent_fitness(self, agent, environment) -> Dict[str, float]:
        """Comprehensive fitness evaluation with dynamic components"""
        if not agent.is_alive:
            return {"total_fitness": 0.0}
            
        # Calculate base fitness components
        base_fitness = self._calculate_base_fitness(agent)
        
        # Apply environmental modifiers
        environmental_fitness = self._apply_environmental_modifiers(agent, base_fitness, environment)
        
        # Competitive fitness adjustment
        competitive_fitness = self._calculate_competitive_fitness(agent, environment)
        
        # Niche specialization bonus
        niche_fitness = self._calculate_niche_fitness(agent)
        
        # Innovation and behavioral complexity
        innovation_fitness = self._calculate_innovation_fitness(agent)
        
        # Combine all components
        total_fitness = self._combine_fitness_components(
            base_fitness, environmental_fitness, competitive_fitness, 
            niche_fitness, innovation_fitness
        )
        
        # Update agent's fitness score
        if hasattr(agent, 'brain'):
            agent.brain.fitness_score = total_fitness["total_fitness"]
        
        return total_fitness
    
    def _calculate_base_fitness(self, agent) -> Dict[str, float]:
        """Calculate base fitness components"""
        fitness = {}
        
        # Survival fitness (age-based with diminishing returns)
        fitness[FitnessObjective.SURVIVAL.value] = math.log(max(1, agent.age)) * 10.0
        
        # Energy efficiency
        if agent.total_distance_traveled > 0:
            efficiency = agent.total_food_consumed / agent.total_distance_traveled
            fitness[FitnessObjective.ENERGY_EFFICIENCY.value] = min(50.0, efficiency * 20.0)
        else:
            fitness[FitnessObjective.ENERGY_EFFICIENCY.value] = agent.energy / agent.max_energy * 10.0
        
        # Reproduction success
        fitness[FitnessObjective.REPRODUCTION.value] = agent.successful_reproductions * 25.0
        
        # Exploration achievement
        unique_areas = len(agent.visited_areas) if agent.visited_areas else 0
        fitness[FitnessObjective.EXPLORATION.value] = min(30.0, unique_areas * 2.0 + agent.novelty_bonus)
        
        # Social cooperation
        social_signals = len(agent.received_signals) if agent.received_signals else 0
        fitness[FitnessObjective.SOCIAL_COOPERATION.value] = min(20.0, social_signals * 1.5)
        
        # Species-specific fitness
        if agent.species_type == SpeciesType.CARNIVORE:
            fitness[FitnessObjective.PREDATION.value] = agent.prey_captures * 18.0
            fitness[FitnessObjective.EVASION.value] = 0.0
        else:
            fitness[FitnessObjective.PREDATION.value] = 0.0
            fitness[FitnessObjective.EVASION.value] = agent.predator_escapes * 15.0
        
        # Innovation (novel behavior patterns)
        fitness[FitnessObjective.INNOVATION.value] = self._measure_behavioral_innovation(agent)
        
        return fitness
    
    def _apply_environmental_modifiers(self, agent, base_fitness: Dict[str, float], 
                                     environment) -> Dict[str, float]:
        """Apply environmental pressure modifiers to fitness"""
        modified_fitness = base_fitness.copy()
        landscape = self.current_landscape
        
        # Apply dynamic weights from current landscape
        for objective, weight in landscape.base_weights.items():
            if objective.value in modified_fitness:
                modified_fitness[objective.value] *= weight
        
        # Resource scarcity modifier
        if landscape.resource_scarcity > 0.5:
            # Penalize energy inefficiency more heavily
            efficiency_penalty = (1.0 - landscape.resource_scarcity) * 0.5
            modified_fitness[FitnessObjective.ENERGY_EFFICIENCY.value] *= (1.0 + efficiency_penalty)
        
        # Population density effects
        if landscape.population_density > 1.0:
            # Increase competition for reproduction
            density_factor = min(0.5, landscape.population_density - 1.0)
            modified_fitness[FitnessObjective.REPRODUCTION.value] *= (1.0 - density_factor)
        
        return modified_fitness
    
    def _calculate_competitive_fitness(self, agent, environment) -> float:
        """Calculate fitness based on competitive performance"""
        if not hasattr(environment, 'agents'):
            return 0.0
            
        same_species_agents = [a for a in environment.agents 
                             if a.is_alive and a.species_type == agent.species_type and a != agent]
        
        if not same_species_agents:
            return 5.0  # Bonus for being the only one of species
        
        # Compare performance against peers
        agent_performance = self._get_agent_performance_score(agent)
        peer_performances = [self._get_agent_performance_score(a) for a in same_species_agents]
        
        if not peer_performances:
            return 5.0
            
        # Rank-based fitness
        better_peers = sum(1 for p in peer_performances if p > agent_performance)
        rank_ratio = 1.0 - (better_peers / len(peer_performances))
        
        # Tournament-style competitive bonus
        competitive_bonus = rank_ratio * 15.0
        
        return competitive_bonus
    
    def _get_agent_performance_score(self, agent) -> float:
        """Get overall performance score for competitive comparison"""
        return (agent.total_food_consumed * 2.0 + 
                agent.successful_reproductions * 10.0 + 
                agent.age * 0.1 + 
                (agent.prey_captures if agent.species_type == SpeciesType.CARNIVORE else agent.predator_escapes) * 5.0)
    
    def _calculate_niche_fitness(self, agent) -> float:
        """Calculate fitness bonus based on ecological niche specialization"""
        # Determine agent's niche assignment
        agent_niche = self._assign_agent_to_niche(agent)
        self.niche_assignments[agent.agent_id] = agent_niche.name
        
        # Calculate how well agent fits their niche
        niche_fit = self._calculate_niche_fit(agent, agent_niche)
        
        # Niche occupancy pressure
        occupancy_pressure = min(1.0, agent_niche.current_occupants / agent_niche.optimal_population_size)
        niche_bonus = niche_fit * (2.0 - occupancy_pressure) * 10.0
        
        return max(0.0, niche_bonus)
    
    def _assign_agent_to_niche(self, agent) -> EcologicalNiche:
        """Assign agent to best-fitting ecological niche"""
        best_niche = None
        best_fit = -1.0
        
        for niche in self.niches.values():
            fit_score = self._calculate_niche_fit(agent, niche)
            if fit_score > best_fit:
                best_fit = fit_score
                best_niche = niche
        
        return best_niche or self.niches["generalist"]
    
    def _calculate_niche_fit(self, agent, niche: EcologicalNiche) -> float:
        """Calculate how well an agent fits a specific niche"""
        # Behavioral trait alignment
        trait_alignment = 0.0
        
        # Approximate behavioral traits from agent performance
        agent_boldness = min(1.0, (agent.total_distance_traveled / max(1, agent.age)) / 2.0)
        agent_cooperation = min(1.0, len(agent.received_signals) / max(1, agent.age / 10)) if agent.received_signals else 0.0
        agent_innovation = min(1.0, agent.novelty_bonus / max(1, agent.age / 20))
        
        # Compare with niche preferences
        trait_alignment += 1.0 - abs(agent_boldness - niche.behavioral_traits["boldness"])
        trait_alignment += 1.0 - abs(agent_cooperation - niche.behavioral_traits["cooperation"])
        trait_alignment += 1.0 - abs(agent_innovation - niche.behavioral_traits["innovation"])
        
        return trait_alignment / 3.0
    
    def _calculate_innovation_fitness(self, agent) -> float:
        """Calculate fitness bonus for innovative behaviors"""
        innovation_score = 0.0
        
        # Novel movement patterns
        if hasattr(agent, 'movement_history') and len(agent.movement_history) > 5:
            movement_diversity = self._calculate_movement_diversity(agent.movement_history)
            innovation_score += movement_diversity * 5.0
        
        # Unique exploration patterns
        if agent.visited_areas and len(agent.visited_areas) > 10:
            exploration_efficiency = len(agent.visited_areas) / max(1, agent.total_distance_traveled / 10)
            innovation_score += min(10.0, exploration_efficiency * 3.0)
        
        # Communication innovation
        if hasattr(agent, 'communication_output') and agent.communication_output > 0:
            communication_innovation = min(5.0, agent.communication_output * 10.0)
            innovation_score += communication_innovation
        
        return innovation_score
    
    def _calculate_movement_diversity(self, movement_history) -> float:
        """Calculate diversity in movement patterns"""
        if len(movement_history) < 3:
            return 0.0
        
        # Calculate variance in movement directions
        directions = []
        for i in range(1, len(movement_history)):
            curr_pos = movement_history[i]['position']
            prev_pos = movement_history[i-1]['position']
            
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            
            if dx != 0 or dy != 0:
                angle = math.atan2(dy, dx)
                directions.append(angle)
        
        if len(directions) < 2:
            return 0.0
        
        # Calculate circular variance for angles
        mean_cos = np.mean([math.cos(angle) for angle in directions])
        mean_sin = np.mean([math.sin(angle) for angle in directions])
        circular_variance = 1.0 - math.sqrt(mean_cos**2 + mean_sin**2)
        
        return circular_variance
    
    def _measure_behavioral_innovation(self, agent) -> float:
        """Measure innovative behaviors and novel strategies"""
        innovation_score = 0.0
        
        # Track unique behavioral combinations
        behavior_key = (
            int(agent.total_distance_traveled / 10),
            int(agent.total_food_consumed),
            agent.successful_reproductions,
            len(agent.visited_areas) if agent.visited_areas else 0
        )
        
        self.innovation_tracking[behavior_key] += 1
        
        # Reward rare behavior patterns
        behavior_rarity = 1.0 / max(1, self.innovation_tracking[behavior_key])
        innovation_score += behavior_rarity * 8.0
        
        return min(15.0, innovation_score)
    
    def _combine_fitness_components(self, base_fitness: Dict[str, float], 
                                  environmental_fitness: Dict[str, float],
                                  competitive_fitness: float, niche_fitness: float,
                                  innovation_fitness: float) -> Dict[str, float]:
        """Combine all fitness components into final score"""
        combined = environmental_fitness.copy()
        
        # Add competitive and niche bonuses
        combined["competitive_bonus"] = competitive_fitness
        combined["niche_specialization"] = niche_fitness
        combined["innovation_bonus"] = innovation_fitness
        
        # Calculate total fitness
        total = sum(combined.values())
        combined["total_fitness"] = total
        
        # Store in history for analysis
        self.fitness_history.append({
            "total": total,
            "timestamp": self.current_landscape.time_step,
            "components": combined.copy()
        })
        
        return combined
    
    def get_fitness_landscape_info(self) -> Dict[str, Any]:
        """Get current fitness landscape information"""
        return {
            "current_weights": dict(self.current_landscape.base_weights),
            "environmental_conditions": {
                "population_density": self.current_landscape.population_density,
                "predation_pressure": self.current_landscape.predation_pressure,
                "resource_scarcity": self.current_landscape.resource_scarcity,
                "competition_pressure": self.current_landscape.competition_pressure
            },
            "active_niches": {name: niche.current_occupants for name, niche in self.niches.items()},
            "recent_fitness_trend": list(self.fitness_history)[-10:] if self.fitness_history else []
        }
    
    def update_niche_populations(self, environment) -> None:
        """Update niche population counts"""
        # Reset counts
        for niche in self.niches.values():
            niche.current_occupants = 0
        
        # Count current assignments
        if hasattr(environment, 'agents'):
            for agent in environment.agents:
                if agent.is_alive and hasattr(agent, 'agent_id'):
                    niche_name = self.niche_assignments.get(agent.agent_id, "generalist")
                    if niche_name in self.niches:
                        self.niches[niche_name].current_occupants += 1
