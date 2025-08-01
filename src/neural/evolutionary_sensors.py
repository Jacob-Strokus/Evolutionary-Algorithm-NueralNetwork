"""
Enhanced Sensory System for Evolutionary Learning
Provides rich environmental information to enable sophisticated behaviors
"""
import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any

class EvolutionarySensorSystem:
    """Enhanced sensory system that provides rich environmental data"""
    
    @staticmethod
    def get_enhanced_sensory_inputs(agent, environment) -> List[float]:
        """
        Extract comprehensive sensory information for evolutionary learning
        Returns 20+ normalized inputs enabling sophisticated behaviors:
        
        INTERNAL STATE (3 inputs):
        [0] Energy level (0-1)
        [1] Age normalized (0-1)  
        [2] Reproduction readiness (0-1)
        
        MULTI-TARGET FOOD SENSING (6 inputs):
        [3] Distance to nearest food (0-1)
        [4] Angle to nearest food (-1 to 1)
        [5] Distance to 2nd nearest food (0-1)
        [6] Angle to 2nd nearest food (-1 to 1)
        [7] Distance to 3rd nearest food (0-1)
        [8] Angle to 3rd nearest food (-1 to 1)
        
        MULTI-TARGET THREAT/PREY SENSING (6 inputs):
        [9] Distance to nearest threat/prey (0-1)
        [10] Angle to nearest threat/prey (-1 to 1)
        [11] Distance to 2nd nearest threat/prey (0-1)
        [12] Angle to 2nd nearest threat/prey (-1 to 1)
        [13] Distance to 3rd nearest threat/prey (0-1)
        [14] Angle to 3rd nearest threat/prey (-1 to 1)
        
        SOCIAL & MOVEMENT SENSING (5 inputs):
        [15] Population density nearby (0-1)
        [16] Average velocity of nearby agents X (-1 to 1)
        [17] Average velocity of nearby agents Y (-1 to 1)
        [18] Communication signal strength (0-1)
        [19] Social attraction/repulsion (-1 to 1)
        
        BOUNDARY & SPATIAL AWARENESS (2 inputs):
        [20] Distance to nearest boundary (0-1)
        [21] Angle to environment center (-1 to 1)
        
        EXPLORATION & MEMORY (3 inputs):
        [22] Area familiarity (0-1) - how often visited
        [23] Time since last food found (0-1)
        [24] Movement efficiency score (0-1)
        """
        inputs = [0.0] * 25
        
        # INTERNAL STATE
        inputs[0] = min(1.0, agent.energy / agent.max_energy)
        inputs[1] = min(1.0, agent.age / 1000.0)
        inputs[2] = 1.0 if agent.can_reproduce() else 0.0
        
        # MULTI-TARGET FOOD SENSING
        food_targets = EvolutionarySensorSystem._find_multiple_food_sources(agent, environment, max_targets=3)
        for i, food_info in enumerate(food_targets):
            base_idx = 3 + i * 2
            inputs[base_idx] = food_info['distance']
            inputs[base_idx + 1] = food_info['angle']
        
        # MULTI-TARGET THREAT/PREY SENSING
        if agent.species_type.value == "herbivore":
            threat_targets = EvolutionarySensorSystem._find_multiple_threats(agent, environment, max_targets=3)
        else:
            threat_targets = EvolutionarySensorSystem._find_multiple_prey(agent, environment, max_targets=3)
        
        for i, target_info in enumerate(threat_targets):
            base_idx = 9 + i * 2
            inputs[base_idx] = target_info['distance']
            inputs[base_idx + 1] = target_info['angle']
        
        # SOCIAL & MOVEMENT SENSING
        social_info = EvolutionarySensorSystem._get_social_information(agent, environment)
        inputs[15] = social_info['density']
        inputs[16] = social_info['avg_velocity_x']
        inputs[17] = social_info['avg_velocity_y']
        inputs[18] = social_info['communication_signal']
        inputs[19] = social_info['social_attraction']
        
        # BOUNDARY & SPATIAL AWARENESS
        boundary_info = EvolutionarySensorSystem._get_boundary_information(agent, environment)
        inputs[20] = boundary_info['nearest_boundary_distance']
        inputs[21] = boundary_info['angle_to_center']
        
        # EXPLORATION & MEMORY
        exploration_info = EvolutionarySensorSystem._get_exploration_information(agent, environment)
        inputs[22] = exploration_info['area_familiarity']
        inputs[23] = exploration_info['time_since_food']
        inputs[24] = exploration_info['movement_efficiency']
        
        return inputs
    
    @staticmethod
    def _find_multiple_food_sources(agent, environment, max_targets=3) -> List[Dict[str, float]]:
        """Find multiple food sources within vision range"""
        food_targets = []
        
        # Get all food sources within vision
        visible_food = []
        for food in environment.food_sources:
            distance = agent.position.distance_to(food.position)
            if distance <= agent.vision_range:
                visible_food.append((food, distance))
        
        # Sort by distance and take closest targets
        visible_food.sort(key=lambda x: x[1])
        
        for i in range(min(max_targets, len(visible_food))):
            food, distance = visible_food[i]
            
            # Calculate angle
            dx = food.position.x - agent.position.x
            dy = food.position.y - agent.position.y
            angle = math.atan2(dy, dx) / math.pi  # Normalize to -1 to 1
            
            food_targets.append({
                'distance': min(1.0, distance / agent.vision_range),
                'angle': angle
            })
        
        # Pad with default values if not enough targets
        while len(food_targets) < max_targets:
            food_targets.append({'distance': 1.0, 'angle': 0.0})
        
        return food_targets
    
    @staticmethod
    def _find_multiple_threats(agent, environment, max_targets=3) -> List[Dict[str, float]]:
        """Find multiple threats within vision range"""
        threat_targets = []
        
        # Get all threats within vision
        visible_threats = []
        for other_agent in environment.agents:
            if (other_agent != agent and other_agent.is_alive and 
                other_agent.species_type.value == "carnivore"):
                distance = agent.position.distance_to(other_agent.position)
                if distance <= agent.vision_range:
                    visible_threats.append((other_agent, distance))
        
        # Sort by distance and take closest threats
        visible_threats.sort(key=lambda x: x[1])
        
        for i in range(min(max_targets, len(visible_threats))):
            threat, distance = visible_threats[i]
            
            # Calculate angle
            dx = threat.position.x - agent.position.x
            dy = threat.position.y - agent.position.y
            angle = math.atan2(dy, dx) / math.pi
            
            threat_targets.append({
                'distance': min(1.0, distance / agent.vision_range),
                'angle': angle
            })
        
        # Pad with default values
        while len(threat_targets) < max_targets:
            threat_targets.append({'distance': 1.0, 'angle': 0.0})
        
        return threat_targets
    
    @staticmethod
    def _find_multiple_prey(agent, environment, max_targets=3) -> List[Dict[str, float]]:
        """Find multiple prey within vision range"""
        prey_targets = []
        
        # Get all prey within vision
        visible_prey = []
        for other_agent in environment.agents:
            if (other_agent != agent and other_agent.is_alive and 
                other_agent.species_type.value == "herbivore"):
                distance = agent.position.distance_to(other_agent.position)
                if distance <= agent.vision_range:
                    visible_prey.append((other_agent, distance))
        
        # Sort by distance and take closest prey
        visible_prey.sort(key=lambda x: x[1])
        
        for i in range(min(max_targets, len(visible_prey))):
            prey, distance = visible_prey[i]
            
            # Calculate angle and consider prey movement
            dx = prey.position.x - agent.position.x
            dy = prey.position.y - agent.position.y
            angle = math.atan2(dy, dx) / math.pi
            
            prey_targets.append({
                'distance': min(1.0, distance / agent.vision_range),
                'angle': angle
            })
        
        # Pad with default values
        while len(prey_targets) < max_targets:
            prey_targets.append({'distance': 1.0, 'angle': 0.0})
        
        return prey_targets
    
    @staticmethod
    def _get_social_information(agent, environment) -> Dict[str, float]:
        """Get information about nearby agents and social dynamics"""
        nearby_agents = []
        total_vx = 0.0
        total_vy = 0.0
        communication_signal = 0.0
        
        for other_agent in environment.agents:
            if other_agent != agent and other_agent.is_alive:
                distance = agent.position.distance_to(other_agent.position)
                if distance <= agent.vision_range:
                    nearby_agents.append(other_agent)
                    
                    # Estimate velocity from position (simplified)
                    # In a real implementation, you'd track actual velocity
                    if hasattr(other_agent, 'last_position'):
                        vx = other_agent.position.x - other_agent.last_position.x
                        vy = other_agent.position.y - other_agent.last_position.y
                        total_vx += vx
                        total_vy += vy
                    
                    # Communication signal (could be evolved behavior)
                    if hasattr(other_agent, 'communication_output'):
                        signal_strength = other_agent.communication_output / max(1.0, distance)
                        communication_signal += signal_strength
        
        num_nearby = len(nearby_agents)
        
        # Calculate social attraction/repulsion
        social_attraction = 0.0
        if num_nearby > 0:
            # Prefer moderate density (not too crowded, not too isolated)
            ideal_density = 0.3
            current_density = min(1.0, num_nearby / 10.0)
            social_attraction = 1.0 - abs(current_density - ideal_density) * 2
            social_attraction = max(-1.0, min(1.0, social_attraction))
        
        return {
            'density': min(1.0, num_nearby / 10.0),
            'avg_velocity_x': np.clip(total_vx / max(1, num_nearby), -1, 1),
            'avg_velocity_y': np.clip(total_vy / max(1, num_nearby), -1, 1),
            'communication_signal': min(1.0, communication_signal),
            'social_attraction': social_attraction
        }
    
    @staticmethod
    def _get_boundary_information(agent, environment) -> Dict[str, float]:
        """Get boundary and spatial awareness information"""
        # Distance to nearest boundary
        dist_to_left = agent.position.x
        dist_to_right = environment.width - agent.position.x
        dist_to_top = agent.position.y
        dist_to_bottom = environment.height - agent.position.y
        
        nearest_boundary_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        max_boundary_dist = min(environment.width, environment.height) / 2
        
        # Angle to environment center
        center_x = environment.width / 2
        center_y = environment.height / 2
        dx = center_x - agent.position.x
        dy = center_y - agent.position.y
        angle_to_center = math.atan2(dy, dx) / math.pi
        
        return {
            'nearest_boundary_distance': min(1.0, nearest_boundary_dist / max_boundary_dist),
            'angle_to_center': angle_to_center
        }
    
    @staticmethod
    def _get_exploration_information(agent, environment) -> Dict[str, float]:
        """Get exploration and memory-related information"""
        # These would be tracked by the agent over time
        # For now, provide simplified versions
        
        # Area familiarity (how often this area has been visited)
        area_familiarity = 0.5  # Default to medium familiarity
        if hasattr(agent, 'visited_areas'):
            current_area = (int(agent.position.x // 10), int(agent.position.y // 10))
            visit_count = agent.visited_areas.get(current_area, 0)
            area_familiarity = min(1.0, visit_count / 10.0)
        
        # Time since last food found
        time_since_food = 0.5  # Default
        if hasattr(agent, 'last_food_time'):
            time_diff = agent.age - agent.last_food_time
            time_since_food = min(1.0, time_diff / 100.0)
        
        # Movement efficiency (distance traveled vs food gained)
        movement_efficiency = 0.5  # Default
        if hasattr(agent, 'total_distance_traveled') and hasattr(agent, 'total_food_consumed'):
            if agent.total_distance_traveled > 0:
                efficiency = agent.total_food_consumed / agent.total_distance_traveled
                movement_efficiency = min(1.0, efficiency)
        
        return {
            'area_familiarity': area_familiarity,
            'time_since_food': time_since_food,
            'movement_efficiency': movement_efficiency
        }
    
    @staticmethod
    def interpret_enhanced_network_output(outputs: np.ndarray) -> Dict[str, Any]:
        """
        Convert enhanced neural network outputs to agent actions
        Enhanced action space with 6 outputs:
        [0] Move X (-1 to 1)
        [1] Move Y (-1 to 1)
        [2] Movement intensity (0-1)
        [3] Reproduce (0/1 threshold at 0.5)
        [4] Communication signal (0-1)
        [5] Exploration bias (0-1) - preference for exploring vs exploiting
        """
        # Normalize outputs
        move_x = (outputs[0] - 0.5) * 2  # Convert from [0,1] to [-1,1]
        move_y = (outputs[1] - 0.5) * 2
        intensity = outputs[2]
        reproduce = outputs[3] > 0.5
        communication = outputs[4]
        exploration_bias = outputs[5]
        
        return {
            'move_x': np.clip(move_x, -1, 1),
            'move_y': np.clip(move_y, -1, 1),
            'intensity': np.clip(intensity, 0, 1),
            'reproduce': reproduce,
            'communication_signal': np.clip(communication, 0, 1),
            'exploration_bias': np.clip(exploration_bias, 0, 1)
        }
