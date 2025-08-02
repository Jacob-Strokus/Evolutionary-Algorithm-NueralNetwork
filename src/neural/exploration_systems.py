"""
Exploration Intelligence System for Phase 2 Evolutionary Neural Networks

This module implements intelligent exploration strategies including:
- Curiosity-driven exploration networks
- Information gain maximization
- Exploration/exploitation balance optimization
- Collective mapping behaviors

Key Features:
- Novelty detection and curiosity-driven behavior
- Information theoretic exploration strategies
- Adaptive exploration based on environmental scarcity
- Collaborative territory mapping and resource discovery
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import math
from collections import defaultdict, deque
import json

class NoveltyDetector:
    """
    Detects novelty in environmental observations for curiosity-driven exploration
    """
    
    def __init__(self, input_size: int = 25, memory_size: int = 1000):
        """
        Initialize novelty detector
        
        Args:
            input_size: Size of sensory input vectors
            memory_size: Size of experience memory buffer
        """
        self.input_size = input_size
        self.memory_size = memory_size
        
        # Experience memory
        self.experience_buffer = deque(maxlen=memory_size)
        self.location_visits = defaultdict(int)
        self.observation_clusters = []
        
        # Novelty assessment parameters
        self.similarity_threshold = 0.8
        self.location_resolution = 5.0  # Grid resolution for location tracking
        self.novelty_decay = 0.99
        
        # Learned representations
        self.cluster_centers = []
        self.cluster_weights = []
        self.max_clusters = 50
        
    def assess_novelty(self, observation: np.ndarray, position: Tuple[float, float]) -> float:
        """
        Assess novelty of current observation and position
        
        Args:
            observation: Current sensory observation vector
            position: Current agent position
            
        Returns:
            Novelty score (0.0 to 1.0, higher = more novel)
        """
        # Spatial novelty (how often has this location been visited)
        spatial_novelty = self._calculate_spatial_novelty(position)
        
        # Observational novelty (how different is this observation)
        observational_novelty = self._calculate_observational_novelty(observation)
        
        # Combined novelty score
        total_novelty = (spatial_novelty * 0.6 + observational_novelty * 0.4)
        
        # Store experience
        self._store_experience(observation, position, total_novelty)
        
        return total_novelty
    
    def _calculate_spatial_novelty(self, position: Tuple[float, float]) -> float:
        """Calculate spatial novelty based on visit history"""
        # Convert position to grid coordinates
        grid_x = int(position[0] / self.location_resolution)
        grid_y = int(position[1] / self.location_resolution)
        grid_key = (grid_x, grid_y)
        
        # Count visits to this grid cell
        visit_count = self.location_visits[grid_key]
        self.location_visits[grid_key] += 1
        
        # Novelty decreases with visit count
        spatial_novelty = 1.0 / (1.0 + visit_count)
        
        return spatial_novelty
    
    def _calculate_observational_novelty(self, observation: np.ndarray) -> float:
        """Calculate observational novelty using clustering"""
        if len(self.cluster_centers) == 0:
            # First observation is maximally novel
            self._add_cluster(observation)
            return 1.0
        
        # Find closest cluster
        distances = []
        for center in self.cluster_centers:
            distance = np.linalg.norm(observation - center)
            distances.append(distance)
        
        min_distance = min(distances)
        closest_idx = distances.index(min_distance)
        
        # Calculate novelty based on distance to closest cluster
        if min_distance > self.similarity_threshold:
            # Novel observation - create new cluster
            if len(self.cluster_centers) < self.max_clusters:
                self._add_cluster(observation)
            else:
                # Replace least weighted cluster
                min_weight_idx = np.argmin(self.cluster_weights)
                self.cluster_centers[min_weight_idx] = observation.copy()
                self.cluster_weights[min_weight_idx] = 1.0
            
            novelty = 0.9  # High novelty for new cluster
        else:
            # Update existing cluster
            learning_rate = 0.1
            self.cluster_centers[closest_idx] = (
                (1 - learning_rate) * self.cluster_centers[closest_idx] +
                learning_rate * observation
            )
            self.cluster_weights[closest_idx] += 1.0
            
            # Novelty inversely related to distance
            novelty = min_distance / self.similarity_threshold
        
        return np.clip(novelty, 0.0, 1.0)
    
    def _add_cluster(self, observation: np.ndarray):
        """Add new cluster center"""
        self.cluster_centers.append(observation.copy())
        self.cluster_weights.append(1.0)
    
    def _store_experience(self, observation: np.ndarray, position: Tuple[float, float], 
                         novelty: float):
        """Store experience in memory buffer"""
        experience = {
            'observation': observation.copy(),
            'position': position,
            'novelty': novelty,
            'timestamp': len(self.experience_buffer)
        }
        
        self.experience_buffer.append(experience)
    
    def get_exploration_map(self) -> Dict:
        """Get exploration map showing visited areas and novelty"""
        exploration_map = {}
        
        for (grid_x, grid_y), visit_count in self.location_visits.items():
            real_x = grid_x * self.location_resolution
            real_y = grid_y * self.location_resolution
            
            exploration_map[(real_x, real_y)] = {
                'visit_count': visit_count,
                'familiarity': min(1.0, visit_count / 10.0),
                'grid_coords': (grid_x, grid_y)
            }
        
        return exploration_map
    
    def suggest_exploration_targets(self, current_pos: Tuple[float, float], 
                                  search_radius: float = 50.0, 
                                  num_targets: int = 3) -> List[Tuple[float, float]]:
        """Suggest exploration targets based on novelty"""
        exploration_map = self.get_exploration_map()
        current_grid_x = int(current_pos[0] / self.location_resolution)
        current_grid_y = int(current_pos[1] / self.location_resolution)
        
        # Generate candidate positions in search radius
        candidates = []
        for dx in range(-int(search_radius / self.location_resolution), 
                       int(search_radius / self.location_resolution) + 1):
            for dy in range(-int(search_radius / self.location_resolution),
                           int(search_radius / self.location_resolution) + 1):
                
                candidate_grid = (current_grid_x + dx, current_grid_y + dy)
                candidate_pos = (candidate_grid[0] * self.location_resolution,
                               candidate_grid[1] * self.location_resolution)
                
                # Calculate distance
                distance = math.sqrt(dx**2 + dy**2) * self.location_resolution
                if distance > search_radius:
                    continue
                
                # Calculate novelty potential
                visit_count = self.location_visits.get(candidate_grid, 0)
                novelty_potential = 1.0 / (1.0 + visit_count)
                
                # Factor in distance (closer targets preferred)
                distance_factor = max(0.1, 1.0 - distance / search_radius)
                
                score = novelty_potential * 0.7 + distance_factor * 0.3
                
                candidates.append((candidate_pos, score))
        
        # Sort by score and return top targets
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [pos for pos, score in candidates[:num_targets]]


class CuriosityDrivenNetwork:
    """
    Neural network that generates curiosity-driven exploration behaviors
    """
    
    def __init__(self, input_size: int = 25, hidden_size: int = 16):
        """
        Initialize curiosity-driven network
        
        Args:
            input_size: Size of input vector (sensory + exploration state)
            hidden_size: Size of hidden layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Network weights
        self.W_input = np.random.normal(0, 0.1, (input_size, hidden_size))
        self.W_hidden = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        self.W_output = np.random.normal(0, 0.1, (hidden_size, 4))  # 4 movement directions
        
        # Biases
        self.b_hidden = np.zeros(hidden_size)
        self.b_output = np.zeros(4)
        
        # Curiosity state
        self.curiosity_level = 0.5
        self.exploration_momentum = np.zeros(2)  # X, Y momentum
        self.recent_novelty = deque(maxlen=10)
        
        # Learning parameters
        self.learning_rate = 0.01
        self.momentum_decay = 0.95
        
    def forward(self, sensory_input: np.ndarray, novelty_signal: float, 
               exploration_state: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass generating curiosity-driven actions
        
        Args:
            sensory_input: Current sensory observation
            novelty_signal: Novelty score from novelty detector
            exploration_state: Current exploration state
            
        Returns:
            action_probabilities: Probabilities for 4 movement directions
            curiosity_info: Information about curiosity processing
        """
        # Update curiosity level based on recent novelty
        self.recent_novelty.append(novelty_signal)
        avg_recent_novelty = np.mean(list(self.recent_novelty))
        
        # Adaptive curiosity level
        if avg_recent_novelty < 0.3:  # Low novelty environment
            self.curiosity_level = min(1.0, self.curiosity_level + 0.05)
        else:  # High novelty environment
            self.curiosity_level = max(0.1, self.curiosity_level - 0.02)
        
        # Prepare enhanced input
        enhanced_input = self._prepare_enhanced_input(
            sensory_input, novelty_signal, exploration_state
        )
        
        # Neural network forward pass
        hidden = np.tanh(np.dot(enhanced_input, self.W_input) + self.b_hidden)
        
        # Apply curiosity-driven modulation
        curiosity_modulation = np.ones(self.hidden_size) * (0.5 + self.curiosity_level * 0.5)
        hidden = hidden * curiosity_modulation
        
        # Recurrent processing
        hidden = np.tanh(np.dot(hidden, self.W_hidden) + hidden)
        
        # Output layer
        output = np.dot(hidden, self.W_output) + self.b_output
        
        # Apply exploration momentum
        momentum_influence = np.array([
            self.exploration_momentum[0],  # Right movement
            -self.exploration_momentum[0], # Left movement  
            self.exploration_momentum[1],  # Up movement
            -self.exploration_momentum[1]  # Down movement
        ]) * 0.3
        
        output += momentum_influence
        
        # Convert to probabilities
        action_probabilities = self._softmax(output)
        
        # Update exploration momentum
        self._update_momentum(action_probabilities, novelty_signal)
        
        # Generate curiosity info
        curiosity_info = {
            'curiosity_level': self.curiosity_level,
            'novelty_signal': novelty_signal,
            'avg_recent_novelty': avg_recent_novelty,
            'exploration_momentum': self.exploration_momentum.copy(),
            'exploration_drive': self._calculate_exploration_drive()
        }
        
        return action_probabilities, curiosity_info
    
    def _prepare_enhanced_input(self, sensory_input: np.ndarray, novelty_signal: float,
                               exploration_state: Dict) -> np.ndarray:
        """Prepare enhanced input with exploration state"""
        enhanced_input = sensory_input.copy()
        
        # Ensure input is correct size
        if len(enhanced_input) < self.input_size:
            # Pad with exploration features
            exploration_features = np.zeros(self.input_size - len(enhanced_input))
            
            # Add exploration-specific features
            if len(exploration_features) > 0:
                exploration_features[0] = novelty_signal
            if len(exploration_features) > 1:
                exploration_features[1] = self.curiosity_level
            if len(exploration_features) > 2:
                exploration_features[2] = exploration_state.get('areas_explored', 0) / 100.0
            if len(exploration_features) > 3:
                exploration_features[3] = exploration_state.get('time_since_discovery', 0) / 50.0
            
            enhanced_input = np.concatenate([enhanced_input, exploration_features])
        
        return enhanced_input[:self.input_size]
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax activation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _update_momentum(self, action_probabilities: np.ndarray, novelty_signal: float):
        """Update exploration momentum based on actions and novelty"""
        # Convert action probabilities to momentum
        right_left = action_probabilities[0] - action_probabilities[1]
        up_down = action_probabilities[2] - action_probabilities[3]
        
        # Update momentum with decay
        self.exploration_momentum[0] = (
            self.exploration_momentum[0] * self.momentum_decay +
            right_left * novelty_signal * 0.1
        )
        self.exploration_momentum[1] = (
            self.exploration_momentum[1] * self.momentum_decay +
            up_down * novelty_signal * 0.1
        )
        
        # Clip momentum
        self.exploration_momentum = np.clip(self.exploration_momentum, -1.0, 1.0)
    
    def _calculate_exploration_drive(self) -> float:
        """Calculate overall exploration drive"""
        momentum_strength = np.linalg.norm(self.exploration_momentum)
        avg_novelty = np.mean(list(self.recent_novelty)) if self.recent_novelty else 0.5
        
        exploration_drive = (
            self.curiosity_level * 0.4 +
            momentum_strength * 0.3 +
            avg_novelty * 0.3
        )
        
        return exploration_drive


class InformationGainCalculator:
    """
    Calculates information gain for different exploration strategies
    """
    
    def __init__(self):
        """Initialize information gain calculator"""
        self.area_information = defaultdict(lambda: {'entropy': 1.0, 'observations': 0})
        self.global_entropy = 1.0
        self.observation_count = 0
        
    def calculate_expected_information_gain(self, target_positions: List[Tuple[float, float]],
                                          current_knowledge: Dict) -> List[float]:
        """
        Calculate expected information gain for target positions
        
        Args:
            target_positions: List of potential exploration targets
            current_knowledge: Current knowledge state
            
        Returns:
            List of information gain estimates for each target
        """
        information_gains = []
        
        for position in target_positions:
            # Get area information
            area_key = self._position_to_area_key(position)
            area_info = self.area_information[area_key]
            
            # Calculate information gain based on:
            # 1. Area entropy (unknown areas have high entropy)
            # 2. Observation density (less observed areas have more potential)
            # 3. Distance from known high-information areas
            
            base_gain = area_info['entropy']
            observation_factor = 1.0 / (1.0 + area_info['observations'])
            distance_factor = self._calculate_distance_factor(position, current_knowledge)
            
            total_gain = base_gain * 0.5 + observation_factor * 0.3 + distance_factor * 0.2
            information_gains.append(total_gain)
        
        return information_gains
    
    def update_information(self, position: Tuple[float, float], 
                          observation: np.ndarray, novelty: float):
        """Update information estimates based on new observation"""
        area_key = self._position_to_area_key(position)
        area_info = self.area_information[area_key]
        
        # Update area entropy based on novelty
        # High novelty = high information content = lower entropy reduction
        entropy_reduction = (1.0 - novelty) * 0.1
        area_info['entropy'] = max(0.1, area_info['entropy'] - entropy_reduction)
        area_info['observations'] += 1
        
        # Update global information measures
        self.observation_count += 1
        self._update_global_entropy()
    
    def _position_to_area_key(self, position: Tuple[float, float]) -> str:
        """Convert position to area key for information tracking"""
        grid_x = int(position[0] / 10.0)  # 10x10 grid
        grid_y = int(position[1] / 10.0)
        return f"{grid_x}_{grid_y}"
    
    def _calculate_distance_factor(self, position: Tuple[float, float], 
                                  knowledge: Dict) -> float:
        """Calculate information gain factor based on distance from known areas"""
        if not knowledge.get('explored_areas', {}):
            return 1.0  # Unknown territory is valuable
        
        min_distance = float('inf')
        for area_pos in knowledge['explored_areas'].keys():
            try:
                # Parse area position
                parts = area_pos.split('_')
                if len(parts) == 2:
                    area_x = float(parts[0]) * 10.0
                    area_y = float(parts[1]) * 10.0
                    distance = math.sqrt((position[0] - area_x)**2 + (position[1] - area_y)**2)
                    min_distance = min(min_distance, distance)
            except:
                continue
        
        if min_distance == float('inf'):
            return 1.0
        
        # Closer to unknown areas = higher information potential
        return min(1.0, min_distance / 50.0)
    
    def _update_global_entropy(self):
        """Update global entropy estimate"""
        if self.observation_count == 0:
            return
        
        # Calculate average area entropy
        area_entropies = [info['entropy'] for info in self.area_information.values()]
        if area_entropies:
            self.global_entropy = np.mean(area_entropies)
    
    def get_information_landscape(self) -> Dict:
        """Get current information landscape"""
        return {
            'area_information': dict(self.area_information),
            'global_entropy': self.global_entropy,
            'total_observations': self.observation_count,
            'explored_areas': len(self.area_information)
        }


class ExplorationIntelligence:
    """
    Main exploration intelligence system combining all exploration components
    """
    
    def __init__(self, agent_id: str, input_size: int = 25):
        """
        Initialize exploration intelligence system
        
        Args:
            agent_id: Unique agent identifier
            input_size: Size of sensory input
        """
        self.agent_id = agent_id
        self.input_size = input_size
        
        # Core components
        self.novelty_detector = NoveltyDetector(input_size)
        self.curiosity_network = CuriosityDrivenNetwork(input_size)
        self.info_gain_calculator = InformationGainCalculator()
        
        # Exploration state
        self.exploration_mode = 'balanced'  # 'balanced', 'exploit', 'explore'
        self.exploration_history = deque(maxlen=100)
        self.resource_discoveries = []
        self.territory_map = {}
        
        # Strategy parameters
        self.exploration_threshold = 0.6
        self.exploitation_threshold = 0.4
        self.strategy_adaptation_rate = 0.05
        
    def process_exploration_decision(self, sensory_input: np.ndarray, 
                                   current_position: Tuple[float, float],
                                   current_knowledge: Dict,
                                   available_resources: List[Dict]) -> Dict:
        """
        Process exploration decision based on current state
        
        Args:
            sensory_input: Current sensory observation
            current_position: Agent's current position
            current_knowledge: Agent's current knowledge
            available_resources: Known resources in environment
            
        Returns:
            Dictionary with exploration decision and reasoning
        """
        # Assess novelty of current situation
        novelty_score = self.novelty_detector.assess_novelty(sensory_input, current_position)
        
        # Get exploration state
        exploration_state = self._build_exploration_state(current_knowledge)
        
        # Generate curiosity-driven actions
        action_probs, curiosity_info = self.curiosity_network.forward(
            sensory_input, novelty_score, exploration_state
        )
        
        # Calculate exploration vs exploitation balance
        exploitation_value = self._assess_exploitation_value(available_resources, current_position)
        exploration_value = self._assess_exploration_value(current_position, current_knowledge)
        
        # Determine strategy
        strategy = self._determine_exploration_strategy(
            exploitation_value, exploration_value, novelty_score
        )
        
        # Generate exploration targets
        exploration_targets = self.novelty_detector.suggest_exploration_targets(
            current_position, search_radius=40.0, num_targets=3
        )
        
        # Calculate information gain for targets
        info_gains = self.info_gain_calculator.calculate_expected_information_gain(
            exploration_targets, current_knowledge
        )
        
        # Select best exploration target
        if exploration_targets and info_gains:
            best_target_idx = np.argmax(info_gains)
            best_target = exploration_targets[best_target_idx]
            best_info_gain = info_gains[best_target_idx]
        else:
            best_target = None
            best_info_gain = 0.0
        
        # Generate decision
        decision = {
            'strategy': strategy,
            'action_probabilities': action_probs,
            'recommended_target': best_target,
            'target_info_gain': best_info_gain,
            'novelty_score': novelty_score,
            'curiosity_info': curiosity_info,
            'exploration_value': exploration_value,
            'exploitation_value': exploitation_value,
            'exploration_targets': list(zip(exploration_targets, info_gains)),
            'reasoning': self._generate_reasoning(strategy, novelty_score, exploitation_value, exploration_value)
        }
        
        # Update exploration history
        self._update_exploration_history(decision, current_position)
        
        # Update information calculator
        self.info_gain_calculator.update_information(current_position, sensory_input, novelty_score)
        
        return decision
    
    def _build_exploration_state(self, knowledge: Dict) -> Dict:
        """Build exploration state from current knowledge"""
        exploration_map = self.novelty_detector.get_exploration_map()
        
        return {
            'areas_explored': len(exploration_map),
            'total_visits': sum(area['visit_count'] for area in exploration_map.values()),
            'time_since_discovery': len(self.exploration_history),
            'known_resources': len(knowledge.get('known_food_sources', [])),
            'known_threats': len(knowledge.get('known_threats', []))
        }
    
    def _assess_exploitation_value(self, available_resources: List[Dict], 
                                  position: Tuple[float, float]) -> float:
        """Assess value of exploiting known resources"""
        if not available_resources:
            return 0.0
        
        total_value = 0.0
        for resource in available_resources:
            # Calculate distance to resource
            distance = math.sqrt(
                (resource['x'] - position[0])**2 + 
                (resource['y'] - position[1])**2
            )
            
            # Value decreases with distance
            resource_value = resource.get('value', 10) / (1.0 + distance / 20.0)
            total_value += resource_value
        
        return min(1.0, total_value / 50.0)  # Normalize
    
    def _assess_exploration_value(self, position: Tuple[float, float], 
                                 knowledge: Dict) -> float:
        """Assess value of exploration"""
        exploration_map = self.novelty_detector.get_exploration_map()
        
        # Base exploration value from unvisited areas nearby
        unvisited_bonus = 0.0
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                check_pos = (position[0] + dx * 5, position[1] + dy * 5)
                if check_pos not in exploration_map:
                    distance = math.sqrt(dx**2 + dy**2) * 5
                    unvisited_bonus += 1.0 / (1.0 + distance / 25.0)
        
        unvisited_bonus = min(1.0, unvisited_bonus / 10.0)
        
        # Recent discovery bonus
        recent_discoveries = len([d for d in self.resource_discoveries[-10:] if d])
        discovery_bonus = min(0.3, recent_discoveries / 10.0)
        
        # Information entropy bonus
        info_landscape = self.info_gain_calculator.get_information_landscape()
        entropy_bonus = info_landscape['global_entropy']
        
        return unvisited_bonus * 0.5 + discovery_bonus * 0.2 + entropy_bonus * 0.3
    
    def _determine_exploration_strategy(self, exploitation_value: float, 
                                       exploration_value: float, 
                                       novelty_score: float) -> str:
        """Determine optimal exploration strategy"""
        
        # Calculate strategy scores
        exploit_score = exploitation_value
        explore_score = exploration_value + novelty_score * 0.3
        
        # Apply current mode bias
        if self.exploration_mode == 'exploit':
            exploit_score *= 1.2
        elif self.exploration_mode == 'explore':
            explore_score *= 1.2
        
        # Determine strategy
        if exploit_score > self.exploitation_threshold and exploit_score > explore_score:
            new_strategy = 'exploit'
        elif explore_score > self.exploration_threshold:
            new_strategy = 'explore'
        else:
            new_strategy = 'balanced'
        
        # Update exploration mode with adaptation
        if new_strategy != self.exploration_mode:
            # Gradual mode transition
            mode_transitions = {
                ('balanced', 'exploit'): 'exploit',
                ('balanced', 'explore'): 'explore', 
                ('exploit', 'balanced'): 'balanced',
                ('explore', 'balanced'): 'balanced',
                ('exploit', 'explore'): 'balanced',  # Transition through balanced
                ('explore', 'exploit'): 'balanced'
            }
            
            transition_key = (self.exploration_mode, new_strategy)
            if transition_key in mode_transitions:
                self.exploration_mode = mode_transitions[transition_key]
            else:
                self.exploration_mode = new_strategy
        
        return new_strategy
    
    def _generate_reasoning(self, strategy: str, novelty: float, 
                           exploitation: float, exploration: float) -> List[str]:
        """Generate reasoning for exploration decision"""
        reasoning = []
        
        reasoning.append(f"Strategy: {strategy}")
        reasoning.append(f"Novelty: {novelty:.3f}")
        reasoning.append(f"Exploitation value: {exploitation:.3f}")
        reasoning.append(f"Exploration value: {exploration:.3f}")
        
        if strategy == 'exploit':
            reasoning.append("Known resources available and valuable")
        elif strategy == 'explore':
            reasoning.append("High exploration potential detected")
        else:
            reasoning.append("Balanced approach optimal")
        
        if novelty > 0.7:
            reasoning.append("High novelty environment - continue exploring")
        elif novelty < 0.3:
            reasoning.append("Familiar environment - consider exploitation")
        
        return reasoning
    
    def _update_exploration_history(self, decision: Dict, position: Tuple[float, float]):
        """Update exploration history"""
        history_entry = {
            'position': position,
            'strategy': decision['strategy'],
            'novelty': decision['novelty_score'],
            'curiosity_level': decision['curiosity_info']['curiosity_level'],
            'timestamp': len(self.exploration_history)
        }
        
        self.exploration_history.append(history_entry)
    
    def update_discoveries(self, new_resources: List[Dict], new_threats: List[Dict]):
        """Update discovered resources and threats"""
        self.resource_discoveries.extend(new_resources)
        
        # Keep recent discoveries only
        if len(self.resource_discoveries) > 50:
            self.resource_discoveries = self.resource_discoveries[-25:]
    
    def get_exploration_stats(self) -> Dict:
        """Get exploration intelligence statistics"""
        exploration_map = self.novelty_detector.get_exploration_map()
        info_landscape = self.info_gain_calculator.get_information_landscape()
        
        # Calculate exploration efficiency
        total_visits = sum(area['visit_count'] for area in exploration_map.values())
        unique_areas = len(exploration_map)
        efficiency = unique_areas / max(1, total_visits)
        
        # Calculate recent discovery rate
        recent_discoveries = len(self.resource_discoveries[-20:])
        discovery_rate = recent_discoveries / 20.0
        
        return {
            'agent_id': self.agent_id,
            'exploration_mode': self.exploration_mode,
            'areas_explored': unique_areas,
            'total_visits': total_visits,
            'exploration_efficiency': efficiency,
            'discovery_rate': discovery_rate,
            'current_curiosity': self.curiosity_network.curiosity_level,
            'global_entropy': info_landscape['global_entropy'],
            'recent_novelty': np.mean(list(self.curiosity_network.recent_novelty)) if self.curiosity_network.recent_novelty else 0.0,
            'exploration_momentum': np.linalg.norm(self.curiosity_network.exploration_momentum)
        }


# Utility functions for integration
def create_exploration_intelligence(agent_id: str, input_size: int = 25) -> ExplorationIntelligence:
    """Factory function to create exploration intelligence system"""
    return ExplorationIntelligence(agent_id, input_size)


def simulate_collective_exploration(exploration_systems: List[ExplorationIntelligence]) -> Dict:
    """
    Simulate collective exploration behavior across multiple agents
    
    Args:
        exploration_systems: List of exploration intelligence systems
        
    Returns:
        Collective exploration metrics
    """
    # Aggregate exploration data
    total_areas = 0
    total_discoveries = 0
    total_efficiency = 0.0
    exploration_coverage = set()
    
    for system in exploration_systems:
        stats = system.get_exploration_stats()
        total_areas += stats['areas_explored']
        total_discoveries += len(system.resource_discoveries)
        total_efficiency += stats['exploration_efficiency']
        
        # Add explored areas to coverage set
        exploration_map = system.novelty_detector.get_exploration_map()
        for pos in exploration_map.keys():
            grid_pos = (int(pos[0] / 10), int(pos[1] / 10))
            exploration_coverage.add(grid_pos)
    
    # Calculate collective metrics
    avg_efficiency = total_efficiency / max(1, len(exploration_systems))
    coverage_area = len(exploration_coverage)
    
    # Calculate exploration diversity
    individual_coverages = []
    for system in exploration_systems:
        exploration_map = system.novelty_detector.get_exploration_map()
        agent_coverage = set()
        for pos in exploration_map.keys():
            grid_pos = (int(pos[0] / 10), int(pos[1] / 10))
            agent_coverage.add(grid_pos)
        individual_coverages.append(agent_coverage)
    
    # Calculate overlap
    if len(individual_coverages) > 1:
        total_overlap = 0
        comparisons = 0
        for i, coverage1 in enumerate(individual_coverages):
            for j, coverage2 in enumerate(individual_coverages[i+1:], i+1):
                overlap = len(coverage1 & coverage2)
                total_coverage = len(coverage1 | coverage2)
                if total_coverage > 0:
                    total_overlap += overlap / total_coverage
                comparisons += 1
        
        avg_overlap = total_overlap / max(1, comparisons)
        exploration_diversity = 1.0 - avg_overlap
    else:
        exploration_diversity = 1.0
    
    return {
        'total_agents': len(exploration_systems),
        'total_areas_explored': total_areas,
        'unique_coverage_area': coverage_area,
        'total_discoveries': total_discoveries,
        'avg_exploration_efficiency': avg_efficiency,
        'exploration_diversity': exploration_diversity,
        'collective_coverage': coverage_area / max(1, len(exploration_systems))
    }
