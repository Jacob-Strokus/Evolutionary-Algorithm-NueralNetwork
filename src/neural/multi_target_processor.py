"""
Multi-Target Decision Fusion System for Phase 2 Evolutionary Neural Networks

This module implements advanced target prioritization and attention-based decision making
for simultaneous processing of multiple food sources and threats.

Key Features:
- Attention-based target prioritization
- Context-aware decision fusion
- Dynamic target switching
- Multi-objective optimization
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import math

class AttentionMechanism:
    """
    Implements attention-based target selection for neural agents
    """
    
    def __init__(self, input_size: int = 25, hidden_size: int = 16):
        """
        Initialize attention mechanism
        
        Args:
            input_size: Size of sensory input vector
            hidden_size: Size of attention hidden layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Attention weights for query, key, value transformations
        self.W_q = np.random.normal(0, 0.1, (input_size, hidden_size))  # Query weights
        self.W_k = np.random.normal(0, 0.1, (input_size, hidden_size))  # Key weights  
        self.W_v = np.random.normal(0, 0.1, (input_size, hidden_size))  # Value weights
        
        # Output projection
        self.W_o = np.random.normal(0, 0.1, (hidden_size, input_size))
        
        # Bias terms
        self.b_q = np.zeros(hidden_size)
        self.b_k = np.zeros(hidden_size)
        self.b_v = np.zeros(hidden_size)
        self.b_o = np.zeros(input_size)
    
    def forward(self, targets: List[np.ndarray], context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply attention mechanism to select and weight targets
        
        Args:
            targets: List of target feature vectors
            context: Current agent context vector
            
        Returns:
            attended_output: Weighted combination of targets
            attention_weights: Attention scores for each target
        """
        if not targets:
            return np.zeros(self.input_size), np.array([])
        
        # Convert targets to matrix
        target_matrix = np.array(targets)  # [num_targets, input_size]
        num_targets = len(targets)
        
        # Compute query from context
        query = np.tanh(np.dot(context, self.W_q) + self.b_q)  # [hidden_size]
        
        # Compute keys and values for all targets
        keys = np.tanh(np.dot(target_matrix, self.W_k) + self.b_k)    # [num_targets, hidden_size]
        values = np.tanh(np.dot(target_matrix, self.W_v) + self.b_v)  # [num_targets, hidden_size]
        
        # Compute attention scores
        scores = np.dot(keys, query)  # [num_targets]
        attention_weights = self._softmax(scores)
        
        # Apply attention weights to values
        attended_values = np.sum(values * attention_weights.reshape(-1, 1), axis=0)  # [hidden_size]
        
        # Project back to input space
        attended_output = np.tanh(np.dot(attended_values, self.W_o) + self.b_o)
        
        return attended_output, attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax activation with numerical stability"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class TargetPrioritizer:
    """
    Implements target prioritization algorithms for multi-target scenarios
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.priority_weights = {
            'distance': 0.4,      # Closer targets preferred
            'value': 0.3,         # Higher value targets preferred  
            'threat_level': 0.2,  # Lower threat preferred
            'novelty': 0.1        # Novel targets slightly preferred
        }
        
        # Tracking for novelty assessment
        self.target_history = {}
        self.encounter_counts = {}
    
    def prioritize_targets(self, targets: List[Dict], agent_pos: Tuple[float, float], 
                          agent_energy: float) -> List[Tuple[Dict, float]]:
        """
        Prioritize targets based on multiple criteria
        
        Args:
            targets: List of target dictionaries with position, type, value
            agent_pos: Current agent position
            agent_energy: Current agent energy level
            
        Returns:
            List of (target, priority_score) tuples sorted by priority
        """
        prioritized = []
        
        for target in targets:
            score = self._calculate_priority_score(target, agent_pos, agent_energy)
            prioritized.append((target, score))
        
        # Sort by priority score (higher is better)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        return prioritized
    
    def _calculate_priority_score(self, target: Dict, agent_pos: Tuple[float, float], 
                                 agent_energy: float) -> float:
        """Calculate priority score for a single target"""
        target_pos = (target['x'], target['y'])
        target_type = target.get('type', 'unknown')
        
        # Distance component (closer is better)
        distance = math.sqrt((target_pos[0] - agent_pos[0])**2 + 
                           (target_pos[1] - agent_pos[1])**2)
        distance_score = 1.0 / (1.0 + distance / 50.0)  # Normalize distance
        
        # Value component
        if target_type == 'food':
            # Higher value for food when energy is low
            energy_multiplier = 2.0 if agent_energy < 50 else 1.0
            value_score = target.get('energy_value', 10) * energy_multiplier / 100.0
        elif target_type == 'threat':
            # Negative value for threats (to be avoided)
            threat_level = target.get('danger_level', 5)
            value_score = -threat_level / 10.0
        else:
            value_score = 0.5  # Neutral value for unknown targets
        
        # Threat level component
        if target_type == 'threat':
            threat_score = -target.get('danger_level', 5) / 10.0
        else:
            threat_score = 0.0
        
        # Novelty component
        target_id = target.get('id', f"{target_pos[0]:.1f},{target_pos[1]:.1f}")
        encounter_count = self.encounter_counts.get(target_id, 0)
        novelty_score = 1.0 / (1.0 + encounter_count)
        
        # Update encounter tracking
        self.encounter_counts[target_id] = encounter_count + 1
        
        # Combine components
        total_score = (
            self.priority_weights['distance'] * distance_score +
            self.priority_weights['value'] * value_score +
            self.priority_weights['threat_level'] * threat_score +
            self.priority_weights['novelty'] * novelty_score
        )
        
        return max(0.0, total_score)  # Ensure non-negative scores


class MultiTargetProcessor:
    """
    Main multi-target processing system combining attention and prioritization
    """
    
    def __init__(self, agent_id: str, input_size: int = 25, max_targets: int = 6):
        """
        Initialize multi-target processor
        
        Args:
            agent_id: Unique identifier for the agent
            input_size: Size of sensory input vectors
            max_targets: Maximum number of targets to process simultaneously
        """
        self.agent_id = agent_id
        self.input_size = input_size
        self.max_targets = max_targets
        
        # Initialize components
        self.attention = AttentionMechanism(input_size)
        self.prioritizer = TargetPrioritizer(agent_id)
        
        # Target type processors
        self.food_processor = AttentionMechanism(input_size, hidden_size=12)
        self.threat_processor = AttentionMechanism(input_size, hidden_size=12)
        
        # Decision fusion weights
        self.fusion_weights = {
            'food_attention': 0.6,
            'threat_attention': 0.4
        }
        
        # Performance tracking
        self.processing_history = []
        self.decision_outcomes = []
    
    def process_targets(self, sensory_input: np.ndarray, detected_targets: List[Dict], 
                       agent_pos: Tuple[float, float], agent_energy: float) -> Dict:
        """
        Process multiple targets and generate decision recommendations
        
        Args:
            sensory_input: Current sensory input vector
            detected_targets: List of detected target dictionaries
            agent_pos: Current agent position  
            agent_energy: Current agent energy level
            
        Returns:
            Dictionary with decision recommendations and attention weights
        """
        # Limit targets to maximum processing capacity
        if len(detected_targets) > self.max_targets:
            # Use prioritizer to select top targets
            prioritized = self.prioritizer.prioritize_targets(
                detected_targets, agent_pos, agent_energy
            )
            detected_targets = [target for target, _ in prioritized[:self.max_targets]]
        
        # Separate targets by type
        food_targets = [t for t in detected_targets if t.get('type') == 'food']
        threat_targets = [t for t in detected_targets if t.get('type') == 'threat']
        
        # Process food targets
        food_features = self._extract_target_features(food_targets, agent_pos)
        food_attention, food_weights = self.food_processor.forward(food_features, sensory_input)
        
        # Process threat targets  
        threat_features = self._extract_target_features(threat_targets, agent_pos)
        threat_attention, threat_weights = self.threat_processor.forward(threat_features, sensory_input)
        
        # Fuse decisions
        decision_vector = self._fuse_decisions(food_attention, threat_attention, 
                                             agent_energy, len(food_targets), len(threat_targets))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            decision_vector, food_targets, threat_targets, 
            food_weights, threat_weights, agent_pos, agent_energy
        )
        
        # Track processing for analysis
        self._record_processing_step(detected_targets, recommendations)
        
        return recommendations
    
    def _extract_target_features(self, targets: List[Dict], agent_pos: Tuple[float, float]) -> List[np.ndarray]:
        """Extract feature vectors for targets"""
        features = []
        
        for target in targets:
            # Create feature vector for target
            target_pos = (target['x'], target['y'])
            distance = math.sqrt((target_pos[0] - agent_pos[0])**2 + 
                               (target_pos[1] - agent_pos[1])**2)
            
            # Direction vector (normalized)
            dx = target_pos[0] - agent_pos[0]
            dy = target_pos[1] - agent_pos[1]
            if distance > 0:
                dx /= distance
                dy /= distance
            
            # Create feature vector
            feature_vector = np.zeros(self.input_size)
            feature_vector[0] = dx                              # X direction
            feature_vector[1] = dy                              # Y direction  
            feature_vector[2] = 1.0 / (1.0 + distance / 50.0)  # Distance (inverse)
            feature_vector[3] = target.get('energy_value', 0) / 100.0  # Value
            feature_vector[4] = target.get('danger_level', 0) / 10.0   # Threat level
            
            # Add type encoding
            if target.get('type') == 'food':
                feature_vector[5] = 1.0
            elif target.get('type') == 'threat':
                feature_vector[6] = 1.0
            
            features.append(feature_vector)
        
        return features
    
    def _fuse_decisions(self, food_attention: np.ndarray, threat_attention: np.ndarray,
                       agent_energy: float, num_food: int, num_threats: int) -> np.ndarray:
        """Fuse food and threat attention into unified decision vector"""
        
        # Adjust fusion weights based on context
        energy_factor = max(0.1, agent_energy / 100.0)  # Lower energy = more food focus
        
        adjusted_food_weight = self.fusion_weights['food_attention'] / energy_factor
        adjusted_threat_weight = self.fusion_weights['threat_attention'] * (1.0 + (num_threats * 0.2))
        
        # Normalize weights
        total_weight = adjusted_food_weight + adjusted_threat_weight
        adjusted_food_weight /= total_weight
        adjusted_threat_weight /= total_weight
        
        # Fuse attention vectors
        decision_vector = (adjusted_food_weight * food_attention + 
                          adjusted_threat_weight * threat_attention)
        
        return decision_vector
    
    def _generate_recommendations(self, decision_vector: np.ndarray, 
                                food_targets: List[Dict], threat_targets: List[Dict],
                                food_weights: np.ndarray, threat_weights: np.ndarray,
                                agent_pos: Tuple[float, float], agent_energy: float) -> Dict:
        """Generate action recommendations based on processed targets"""
        
        recommendations = {
            'decision_vector': decision_vector,
            'primary_action': 'explore',  # Default action
            'target_food': None,
            'avoid_threat': None,
            'confidence': 0.0,
            'reasoning': [],
            'attention_weights': {
                'food': food_weights.tolist() if len(food_weights) > 0 else [],
                'threat': threat_weights.tolist() if len(threat_weights) > 0 else []
            }
        }
        
        # Determine primary action based on strongest signal
        max_signal = np.max(np.abs(decision_vector))
        confidence = min(1.0, max_signal * 2.0)
        recommendations['confidence'] = confidence
        
        # Food targeting logic
        if len(food_targets) > 0 and len(food_weights) > 0:
            best_food_idx = np.argmax(food_weights)
            best_food = food_targets[best_food_idx]
            food_confidence = food_weights[best_food_idx]
            
            if food_confidence > 0.3:  # Threshold for food targeting
                recommendations['primary_action'] = 'target_food'
                recommendations['target_food'] = best_food
                recommendations['reasoning'].append(f"High food attention: {food_confidence:.2f}")
        
        # Threat avoidance logic
        if len(threat_targets) > 0 and len(threat_weights) > 0:
            worst_threat_idx = np.argmax(threat_weights)  # Highest attention = most dangerous
            worst_threat = threat_targets[worst_threat_idx]
            threat_attention = threat_weights[worst_threat_idx]
            
            if threat_attention > 0.4:  # Threshold for threat avoidance
                recommendations['primary_action'] = 'avoid_threat'
                recommendations['avoid_threat'] = worst_threat
                recommendations['reasoning'].append(f"High threat attention: {threat_attention:.2f}")
        
        # Energy-based decision modulation
        if agent_energy < 30 and len(food_targets) > 0:
            recommendations['primary_action'] = 'target_food'
            recommendations['reasoning'].append(f"Low energy: {agent_energy}")
        
        return recommendations
    
    def _record_processing_step(self, targets: List[Dict], recommendations: Dict):
        """Record processing step for analysis and learning"""
        step_record = {
            'timestamp': len(self.processing_history),
            'num_targets': len(targets),
            'action': recommendations['primary_action'],
            'confidence': recommendations['confidence'],
            'reasoning_count': len(recommendations['reasoning'])
        }
        
        self.processing_history.append(step_record)
        
        # Maintain history size
        if len(self.processing_history) > 1000:
            self.processing_history = self.processing_history[-500:]
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about target processing performance"""
        if not self.processing_history:
            return {'total_steps': 0}
        
        recent_history = self.processing_history[-100:]  # Last 100 steps
        
        stats = {
            'total_steps': len(self.processing_history),
            'recent_avg_targets': np.mean([s['num_targets'] for s in recent_history]),
            'recent_avg_confidence': np.mean([s['confidence'] for s in recent_history]),
            'action_distribution': {},
            'reasoning_complexity': np.mean([s['reasoning_count'] for s in recent_history])
        }
        
        # Action distribution
        for action in ['explore', 'target_food', 'avoid_threat']:
            count = sum(1 for s in recent_history if s['action'] == action)
            stats['action_distribution'][action] = count / len(recent_history)
        
        return stats


# Utility functions for integration
def create_multi_target_processor(agent_id: str, sensory_input_size: int = 25) -> MultiTargetProcessor:
    """Factory function to create a multi-target processor"""
    return MultiTargetProcessor(agent_id, sensory_input_size)


def integrate_with_neural_network(processor: MultiTargetProcessor, 
                                 neural_network, sensory_input: np.ndarray) -> np.ndarray:
    """
    Integrate multi-target processing with existing neural network
    
    Args:
        processor: MultiTargetProcessor instance
        neural_network: Existing neural network
        sensory_input: Raw sensory input
        
    Returns:
        Enhanced input vector for neural network
    """
    # This will be implemented when integrating with existing neural networks
    # For now, return the enhanced sensory input
    return sensory_input
