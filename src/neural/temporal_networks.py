"""
Advanced Recurrent Networks for Phase 2 Evolutionary Neural Networks

This module implements sophisticated temporal learning patterns including:
- LSTM-style gated recurrent units
- Multi-timescale memory systems  
- Sequence learning for complex behaviors
- Temporal pattern recognition

Key Features:
- Gated memory cells with forget/input/output gates
- Multi-timescale processing (short-term, medium-term, long-term)
- Sequence prediction and pattern recognition
- Temporal attention mechanisms
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import math

class GatedMemoryCell:
    """
    LSTM-style gated memory cell for temporal learning
    """
    
    def __init__(self, input_size: int, hidden_size: int, cell_id: str = "default"):
        """
        Initialize gated memory cell
        
        Args:
            input_size: Size of input vector
            hidden_size: Size of hidden state
            cell_id: Unique identifier for this cell
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_id = cell_id
        
        # Initialize gate weights (forget, input, output)
        self.W_f = np.random.normal(0, 0.1, (input_size + hidden_size, hidden_size))  # Forget gate
        self.W_i = np.random.normal(0, 0.1, (input_size + hidden_size, hidden_size))  # Input gate
        self.W_o = np.random.normal(0, 0.1, (input_size + hidden_size, hidden_size))  # Output gate
        self.W_c = np.random.normal(0, 0.1, (input_size + hidden_size, hidden_size))  # Cell candidate
        
        # Bias terms
        self.b_f = np.zeros(hidden_size)
        self.b_i = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)
        self.b_c = np.zeros(hidden_size)
        
        # Memory state
        self.cell_state = np.zeros(hidden_size)
        self.hidden_state = np.zeros(hidden_size)
        
        # For sequence learning
        self.sequence_memory = []
        self.max_sequence_length = 50
        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through gated memory cell
        
        Args:
            x: Input vector
            
        Returns:
            hidden_state: Current hidden state
            cell_state: Current cell state
        """
        # Concatenate input and previous hidden state
        combined = np.concatenate([x, self.hidden_state])
        
        # Compute gates
        forget_gate = self._sigmoid(np.dot(combined, self.W_f) + self.b_f)
        input_gate = self._sigmoid(np.dot(combined, self.W_i) + self.b_i)
        output_gate = self._sigmoid(np.dot(combined, self.W_o) + self.b_o)
        
        # Compute cell candidate
        cell_candidate = np.tanh(np.dot(combined, self.W_c) + self.b_c)
        
        # Update cell state
        self.cell_state = forget_gate * self.cell_state + input_gate * cell_candidate
        
        # Update hidden state
        self.hidden_state = output_gate * np.tanh(self.cell_state)
        
        # Store in sequence memory
        self._update_sequence_memory(x, self.hidden_state)
        
        return self.hidden_state.copy(), self.cell_state.copy()
    
    def reset_state(self):
        """Reset cell and hidden states"""
        self.cell_state = np.zeros(self.hidden_size)
        self.hidden_state = np.zeros(self.hidden_size)
        self.sequence_memory.clear()
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation with numerical stability"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _update_sequence_memory(self, input_vec: np.ndarray, hidden_vec: np.ndarray):
        """Update sequence memory for pattern learning"""
        self.sequence_memory.append({
            'input': input_vec.copy(),
            'hidden': hidden_vec.copy(),
            'timestamp': len(self.sequence_memory)
        })
        
        # Maintain memory size
        if len(self.sequence_memory) > self.max_sequence_length:
            self.sequence_memory.pop(0)
    
    def get_sequence_patterns(self) -> List[Dict]:
        """Extract learned sequence patterns"""
        if len(self.sequence_memory) < 3:
            return []
        
        patterns = []
        
        # Look for repeating subsequences
        for start_idx in range(len(self.sequence_memory) - 2):
            for length in range(2, min(6, len(self.sequence_memory) - start_idx)):
                pattern = self.sequence_memory[start_idx:start_idx + length]
                
                # Check if this pattern repeats later
                pattern_found = False
                for check_idx in range(start_idx + length, len(self.sequence_memory) - length + 1):
                    if self._patterns_match(pattern, self.sequence_memory[check_idx:check_idx + length]):
                        pattern_found = True
                        break
                
                if pattern_found:
                    patterns.append({
                        'pattern': pattern,
                        'length': length,
                        'first_occurrence': start_idx,
                        'confidence': self._calculate_pattern_confidence(pattern)
                    })
        
        return patterns
    
    def _patterns_match(self, pattern1: List[Dict], pattern2: List[Dict], threshold: float = 0.8) -> bool:
        """Check if two patterns match within threshold"""
        if len(pattern1) != len(pattern2):
            return False
        
        similarities = []
        for p1, p2 in zip(pattern1, pattern2):
            sim = self._cosine_similarity(p1['hidden'], p2['hidden'])
            similarities.append(sim)
        
        return np.mean(similarities) > threshold
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_pattern_confidence(self, pattern: List[Dict]) -> float:
        """Calculate confidence score for a pattern"""
        # Base confidence on pattern length and hidden state consistency
        length_bonus = min(1.0, len(pattern) / 5.0)
        
        # Measure consistency of hidden states in pattern
        hidden_states = [p['hidden'] for p in pattern]
        consistency = 0.0
        
        if len(hidden_states) > 1:
            similarities = []
            for i in range(len(hidden_states) - 1):
                sim = self._cosine_similarity(hidden_states[i], hidden_states[i + 1])
                similarities.append(sim)
            consistency = np.mean(similarities)
        
        return (length_bonus + consistency) / 2.0


class MultiTimescaleMemory:
    """
    Multi-timescale memory system for different temporal ranges
    """
    
    def __init__(self, input_size: int):
        """
        Initialize multi-timescale memory
        
        Args:
            input_size: Size of input vectors
        """
        self.input_size = input_size
        
        # Different timescale memory cells
        self.short_term = GatedMemoryCell(input_size, 16, "short_term")    # ~5-10 steps
        self.medium_term = GatedMemoryCell(input_size, 24, "medium_term")  # ~20-50 steps  
        self.long_term = GatedMemoryCell(input_size, 32, "long_term")      # ~100+ steps
        
        # Timescale tracking
        self.step_count = 0
        self.medium_update_interval = 3
        self.long_update_interval = 10
        
        # Integration weights
        self.integration_weights = np.array([0.5, 0.3, 0.2])  # short, medium, long
        
    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Process input through multi-timescale memory
        
        Args:
            x: Input vector
            
        Returns:
            Integrated memory output
        """
        self.step_count += 1
        
        # Always update short-term memory
        short_hidden, _ = self.short_term.forward(x)
        
        # Update medium-term memory at intervals
        medium_hidden = self.medium_term.hidden_state.copy()
        if self.step_count % self.medium_update_interval == 0:
            medium_hidden, _ = self.medium_term.forward(x)
        
        # Update long-term memory at longer intervals
        long_hidden = self.long_term.hidden_state.copy()
        if self.step_count % self.long_update_interval == 0:
            long_hidden, _ = self.long_term.forward(x)
        
        # Integrate memories
        # Pad shorter vectors to match longest
        max_size = max(len(short_hidden), len(medium_hidden), len(long_hidden))
        
        short_padded = np.pad(short_hidden, (0, max_size - len(short_hidden)))
        medium_padded = np.pad(medium_hidden, (0, max_size - len(medium_hidden)))
        long_padded = np.pad(long_hidden, (0, max_size - len(long_hidden)))
        
        # Weighted combination
        integrated = (self.integration_weights[0] * short_padded +
                     self.integration_weights[1] * medium_padded +
                     self.integration_weights[2] * long_padded)
        
        return integrated
    
    def get_all_patterns(self) -> Dict[str, List[Dict]]:
        """Get patterns from all timescale memories"""
        return {
            'short_term': self.short_term.get_sequence_patterns(),
            'medium_term': self.medium_term.get_sequence_patterns(),
            'long_term': self.long_term.get_sequence_patterns()
        }
    
    def reset_all(self):
        """Reset all memory timescales"""
        self.short_term.reset_state()
        self.medium_term.reset_state()
        self.long_term.reset_state()
        self.step_count = 0


class TemporalPatternRecognizer:
    """
    Recognizes and learns temporal patterns in behavior sequences
    """
    
    def __init__(self, pattern_length: int = 5):
        """
        Initialize temporal pattern recognizer
        
        Args:
            pattern_length: Length of patterns to recognize
        """
        self.pattern_length = pattern_length
        self.learned_patterns = {}
        self.recent_sequence = []
        self.max_recent_length = 100
        
        # Pattern statistics
        self.pattern_frequencies = {}
        self.pattern_rewards = {}
        
    def observe_action(self, action: str, reward: float = 0.0):
        """
        Observe an action and its reward for pattern learning
        
        Args:
            action: Action taken
            reward: Reward received for the action
        """
        # Add to recent sequence
        self.recent_sequence.append({
            'action': action,
            'reward': reward,
            'timestamp': len(self.recent_sequence)
        })
        
        # Maintain sequence size
        if len(self.recent_sequence) > self.max_recent_length:
            self.recent_sequence.pop(0)
        
        # Extract patterns if we have enough data
        if len(self.recent_sequence) >= self.pattern_length:
            self._extract_patterns()
    
    def _extract_patterns(self):
        """Extract patterns from recent sequence"""
        # Look for patterns of specified length
        for i in range(len(self.recent_sequence) - self.pattern_length + 1):
            pattern = tuple([s['action'] for s in 
                           self.recent_sequence[i:i + self.pattern_length]])
            
            # Calculate pattern reward
            pattern_rewards = [s['reward'] for s in 
                             self.recent_sequence[i:i + self.pattern_length]]
            avg_reward = np.mean(pattern_rewards)
            
            # Update pattern tracking
            if pattern not in self.learned_patterns:
                self.learned_patterns[pattern] = {
                    'frequency': 0,
                    'total_reward': 0.0,
                    'avg_reward': 0.0,
                    'last_seen': 0
                }
            
            # Update statistics
            self.learned_patterns[pattern]['frequency'] += 1
            self.learned_patterns[pattern]['total_reward'] += avg_reward
            self.learned_patterns[pattern]['avg_reward'] = (
                self.learned_patterns[pattern]['total_reward'] / 
                self.learned_patterns[pattern]['frequency']
            )
            self.learned_patterns[pattern]['last_seen'] = len(self.recent_sequence)
    
    def predict_next_action(self, current_sequence: List[str]) -> Dict[str, float]:
        """
        Predict next action based on learned patterns
        
        Args:
            current_sequence: Recent action sequence
            
        Returns:
            Dictionary of action predictions with confidence scores
        """
        if len(current_sequence) < self.pattern_length - 1:
            return {}
        
        # Look for matching pattern prefixes
        prefix = tuple(current_sequence[-(self.pattern_length - 1):])
        predictions = {}
        
        for pattern, stats in self.learned_patterns.items():
            if pattern[:-1] == prefix:  # Pattern matches prefix
                next_action = pattern[-1]
                confidence = self._calculate_prediction_confidence(stats)
                
                if next_action not in predictions:
                    predictions[next_action] = 0.0
                predictions[next_action] = max(predictions[next_action], confidence)
        
        return predictions
    
    def _calculate_prediction_confidence(self, pattern_stats: Dict) -> float:
        """Calculate confidence for a pattern prediction"""
        # Base confidence on frequency and reward
        frequency_score = min(1.0, pattern_stats['frequency'] / 10.0)
        reward_score = max(0.0, min(1.0, (pattern_stats['avg_reward'] + 1.0) / 2.0))
        
        # Decay based on recency
        recency_factor = max(0.1, 1.0 - (len(self.recent_sequence) - 
                                        pattern_stats['last_seen']) / 50.0)
        
        return (frequency_score + reward_score) * recency_factor / 2.0
    
    def get_best_patterns(self, min_frequency: int = 2) -> List[Dict]:
        """Get best learned patterns above frequency threshold"""
        good_patterns = []
        
        for pattern, stats in self.learned_patterns.items():
            if stats['frequency'] >= min_frequency:
                good_patterns.append({
                    'pattern': pattern,
                    'frequency': stats['frequency'],
                    'avg_reward': stats['avg_reward'],
                    'confidence': self._calculate_prediction_confidence(stats)
                })
        
        # Sort by confidence
        good_patterns.sort(key=lambda x: x['confidence'], reverse=True)
        return good_patterns


class AdvancedRecurrentNetwork:
    """
    Main advanced recurrent network combining all temporal components
    """
    
    def __init__(self, input_size: int, output_size: int, agent_id: str = "default"):
        """
        Initialize advanced recurrent network
        
        Args:
            input_size: Size of input vectors
            output_size: Size of output vectors
            agent_id: Unique identifier for this network
        """
        self.input_size = input_size
        self.output_size = output_size
        self.agent_id = agent_id
        
        # Core components
        self.multi_memory = MultiTimescaleMemory(input_size)
        self.pattern_recognizer = TemporalPatternRecognizer()
        
        # Output projection layer
        memory_size = 32  # From long-term memory
        self.output_weights = np.random.normal(0, 0.1, (memory_size, output_size))
        self.output_bias = np.zeros(output_size)
        
        # Context integration
        self.context_weights = np.random.normal(0, 0.1, (input_size, memory_size))
        
        # Performance tracking
        self.processing_history = []
        self.prediction_accuracy = []
        
    def forward(self, x: np.ndarray, current_action: Optional[str] = None, 
               reward: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through advanced recurrent network
        
        Args:
            x: Input vector
            current_action: Current action being taken (for pattern learning)
            reward: Reward for current action
            
        Returns:
            output: Network output
            temporal_info: Dictionary with temporal processing information
        """
        # Process through multi-timescale memory
        memory_output = self.multi_memory.process(x)
        
        # Context integration
        context = np.tanh(np.dot(x, self.context_weights))
        
        # Combine memory and context
        combined = memory_output + context
        
        # Generate output
        output = np.tanh(np.dot(combined, self.output_weights) + self.output_bias)
        
        # Update pattern recognition if action provided
        if current_action is not None:
            self.pattern_recognizer.observe_action(current_action, reward)
        
        # Get temporal information
        temporal_info = self._extract_temporal_info(memory_output)
        
        # Track processing
        self._record_processing_step(x, output, temporal_info)
        
        return output, temporal_info
    
    def predict_next_actions(self, recent_actions: List[str]) -> Dict[str, float]:
        """Get predictions for next actions based on learned patterns"""
        return self.pattern_recognizer.predict_next_action(recent_actions)
    
    def _extract_temporal_info(self, memory_output: np.ndarray) -> Dict:
        """Extract temporal processing information"""
        # Get patterns from all timescales
        all_patterns = self.multi_memory.get_all_patterns()
        
        # Get best learned behavioral patterns
        behavioral_patterns = self.pattern_recognizer.get_best_patterns()
        
        # Memory activation analysis
        memory_activation = {
            'strength': np.linalg.norm(memory_output),
            'distribution': np.histogram(memory_output, bins=5)[0].tolist(),
            'dominant_features': np.argsort(np.abs(memory_output))[-3:].tolist()
        }
        
        return {
            'neural_patterns': all_patterns,
            'behavioral_patterns': behavioral_patterns[:5],  # Top 5
            'memory_activation': memory_activation,
            'timescale_step': self.multi_memory.step_count
        }
    
    def _record_processing_step(self, input_vec: np.ndarray, output_vec: np.ndarray, 
                               temporal_info: Dict):
        """Record processing step for analysis"""
        step_record = {
            'step': len(self.processing_history),
            'input_strength': np.linalg.norm(input_vec),
            'output_strength': np.linalg.norm(output_vec),
            'num_patterns': len(temporal_info['behavioral_patterns']),
            'memory_strength': temporal_info['memory_activation']['strength']
        }
        
        self.processing_history.append(step_record)
        
        # Maintain history size
        if len(self.processing_history) > 500:
            self.processing_history = self.processing_history[-250:]
    
    def reset_memory(self):
        """Reset all memory components"""
        self.multi_memory.reset_all()
        self.pattern_recognizer.recent_sequence.clear()
    
    def get_temporal_stats(self) -> Dict:
        """Get statistics about temporal processing"""
        if not self.processing_history:
            return {'total_steps': 0}
        
        recent = self.processing_history[-50:]  # Last 50 steps
        
        stats = {
            'total_steps': len(self.processing_history),
            'avg_memory_strength': np.mean([s['memory_strength'] for s in recent]),
            'avg_pattern_count': np.mean([s['num_patterns'] for s in recent]),
            'memory_stability': np.std([s['memory_strength'] for s in recent]),
            'learned_patterns': len(self.pattern_recognizer.learned_patterns),
            'pattern_quality': self._assess_pattern_quality()
        }
        
        return stats
    
    def _assess_pattern_quality(self) -> float:
        """Assess quality of learned patterns"""
        if not self.pattern_recognizer.learned_patterns:
            return 0.0
        
        qualities = []
        for pattern, stats in self.pattern_recognizer.learned_patterns.items():
            # Quality based on frequency and reward
            freq_score = min(1.0, stats['frequency'] / 5.0)
            reward_score = max(0.0, (stats['avg_reward'] + 1.0) / 2.0)
            qualities.append((freq_score + reward_score) / 2.0)
        
        return np.mean(qualities)


# Utility functions for integration
def create_advanced_recurrent_network(input_size: int, output_size: int, 
                                    agent_id: str) -> AdvancedRecurrentNetwork:
    """Factory function to create advanced recurrent network"""
    return AdvancedRecurrentNetwork(input_size, output_size, agent_id)


def integrate_temporal_processing(network: AdvancedRecurrentNetwork, 
                                base_neural_network, 
                                sensory_input: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Integrate temporal processing with base neural network
    
    Args:
        network: AdvancedRecurrentNetwork instance
        base_neural_network: Base neural network
        sensory_input: Current sensory input
        
    Returns:
        enhanced_output: Enhanced network output with temporal processing
        temporal_info: Temporal processing information
    """
    # Process through temporal network
    temporal_output, temporal_info = network.forward(sensory_input)
    
    # This will be fully implemented during integration phase
    # For now, return temporal output and info
    return temporal_output, temporal_info
