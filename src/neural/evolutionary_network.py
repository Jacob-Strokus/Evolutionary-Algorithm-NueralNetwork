"""
Evolutionary Neural Network System
Designed for evolution-driven learning rather than design-driven constraints
"""
import numpy as np
import random
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque

@dataclass
class EvolutionaryNetworkConfig:
    """Configuration for evolutionary neural network architecture"""
    min_input_size: int = 20  # Rich sensory inputs
    max_input_size: int = 25  # Expandable input space
    min_hidden_size: int = 8   # Minimum hidden neurons
    max_hidden_size: int = 32  # Maximum hidden neurons (evolved trait)
    output_size: int = 6       # Enhanced action space
    memory_size: int = 3       # Short-term memory cells
    max_targets: int = 3       # Multiple target sensing
    mutation_rate: float = 0.15
    mutation_strength: float = 0.3
    structure_mutation_rate: float = 0.05  # Chance to change network size
    recurrent_probability: float = 0.3     # Chance for recurrent connections

class EvolutionaryNeuralNetwork:
    """Enhanced neural network designed for evolutionary learning"""
    
    def __init__(self, config: EvolutionaryNetworkConfig):
        self.config = config
        
        # Evolve network structure
        self.hidden_size = random.randint(config.min_hidden_size, config.max_hidden_size)
        self.has_recurrent = random.random() < config.recurrent_probability
        
        # Initialize weights
        self.weights_input_hidden = np.random.randn(config.min_input_size, self.hidden_size) * 0.4
        self.bias_hidden = np.random.randn(self.hidden_size) * 0.2
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, config.output_size) * 0.4
        self.bias_output = np.random.randn(config.output_size) * 0.2
        
        # Recurrent connections (hidden to hidden)
        if self.has_recurrent:
            self.weights_recurrent = np.random.randn(self.hidden_size, self.hidden_size) * 0.2
        else:
            self.weights_recurrent = None
        
        # Memory system
        self.memory = deque(maxlen=config.memory_size)
        self.hidden_state = np.zeros(self.hidden_size)
        
        # Performance tracking
        self.fitness_score = 0.0
        self.age = 0
        self.decisions_made = 0
        self.exploration_bonus = 0.0
        self.social_interactions = 0
        
        # Learning traits (evolved)
        self.exploration_drive = random.random()  # 0-1: how much to explore vs exploit
        self.social_weight = random.random()      # 0-1: how much to consider other agents
        self.memory_decay = 0.8 + random.random() * 0.2  # How quickly memory fades
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(np.clip(x, -500, 500))
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def forward(self, inputs: np.ndarray, environment_state: Dict[str, Any] = None) -> np.ndarray:
        """Enhanced forward pass with memory and recurrent connections"""
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        
        # Add memory context to inputs
        memory_context = self._get_memory_context()
        enhanced_inputs = np.concatenate([inputs[:self.config.min_input_size], memory_context])
        
        # Ensure input size matches network
        if enhanced_inputs.shape[0] != self.weights_input_hidden.shape[0]:
            # Pad or truncate to match
            target_size = self.weights_input_hidden.shape[0]
            if enhanced_inputs.shape[0] < target_size:
                enhanced_inputs = np.pad(enhanced_inputs, (0, target_size - enhanced_inputs.shape[0]))
            else:
                enhanced_inputs = enhanced_inputs[:target_size]
        
        # Input to hidden layer
        hidden_input = np.dot(enhanced_inputs, self.weights_input_hidden) + self.bias_hidden
        
        # Add recurrent connections if present
        if self.has_recurrent and self.weights_recurrent is not None:
            recurrent_input = np.dot(self.hidden_state, self.weights_recurrent)
            hidden_input += recurrent_input
        
        # Apply activation
        hidden_output = self.tanh(hidden_input)
        
        # Update hidden state for next time step
        self.hidden_state = hidden_output * self.memory_decay
        
        # Hidden to output layer
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.sigmoid(output_input)
        
        # Store experience in memory
        self._update_memory(enhanced_inputs, hidden_output, output)
        
        self.decisions_made += 1
        return output
    
    def _get_memory_context(self) -> np.ndarray:
        """Extract context from recent memory"""
        if not self.memory:
            return np.zeros(self.config.memory_size * 2)  # Input + output context
        
        # Combine recent inputs and outputs
        context = []
        for mem in self.memory:
            # Use average of recent inputs and outputs as context
            input_summary = np.mean(mem['input'][:3]) if len(mem['input']) > 3 else 0.0
            output_summary = np.mean(mem['output'][:2]) if len(mem['output']) > 2 else 0.0
            context.extend([input_summary, output_summary])
        
        # Pad if necessary
        while len(context) < self.config.memory_size * 2:
            context.extend([0.0, 0.0])
        
        return np.array(context[:self.config.memory_size * 2])
    
    def _update_memory(self, inputs: np.ndarray, hidden: np.ndarray, outputs: np.ndarray):
        """Update short-term memory with recent experience"""
        memory_entry = {
            'input': inputs,
            'hidden': hidden,
            'output': outputs,
            'timestamp': self.decisions_made
        }
        self.memory.append(memory_entry)
    
    def mutate(self):
        """Enhanced mutation with structural changes"""
        mutation_rate = self.config.mutation_rate
        mutation_strength = self.config.mutation_strength
        
        # Standard weight mutations
        self._mutate_weights(self.weights_input_hidden, mutation_rate, mutation_strength)
        self._mutate_weights(self.weights_hidden_output, mutation_rate, mutation_strength)
        self._mutate_weights(self.bias_hidden, mutation_rate, mutation_strength)
        self._mutate_weights(self.bias_output, mutation_rate, mutation_strength)
        
        # Recurrent connection mutations
        if self.has_recurrent and self.weights_recurrent is not None:
            self._mutate_weights(self.weights_recurrent, mutation_rate, mutation_strength * 0.5)
        
        # Mutate behavioral traits
        if random.random() < mutation_rate:
            self.exploration_drive += np.random.normal(0, 0.1)
            self.exploration_drive = np.clip(self.exploration_drive, 0, 1)
        
        if random.random() < mutation_rate:
            self.social_weight += np.random.normal(0, 0.1)
            self.social_weight = np.clip(self.social_weight, 0, 1)
        
        if random.random() < mutation_rate:
            self.memory_decay += np.random.normal(0, 0.05)
            self.memory_decay = np.clip(self.memory_decay, 0.5, 1.0)
        
        # Structural mutations (rare)
        if random.random() < self.config.structure_mutation_rate:
            self._mutate_structure()
    
    def _mutate_weights(self, weights, rate, strength):
        """Apply mutations to weight arrays"""
        if weights is None:
            return
        mutation_mask = np.random.random(weights.shape) < rate
        mutations = np.random.randn(*weights.shape) * strength
        weights += mutation_mask * mutations
    
    def _mutate_structure(self):
        """Mutate network structure (add/remove neurons or connections)"""
        # Change hidden layer size
        if random.random() < 0.3:
            old_size = self.hidden_size
            self.hidden_size = random.randint(self.config.min_hidden_size, self.config.max_hidden_size)
            
            if self.hidden_size != old_size:
                # Resize weight matrices
                self._resize_hidden_layer(old_size, self.hidden_size)
        
        # Add/remove recurrent connections
        if random.random() < 0.2:
            if self.has_recurrent and random.random() < 0.5:
                # Remove recurrent connections
                self.has_recurrent = False
                self.weights_recurrent = None
            elif not self.has_recurrent:
                # Add recurrent connections
                self.has_recurrent = True
                self.weights_recurrent = np.random.randn(self.hidden_size, self.hidden_size) * 0.2
    
    def _resize_hidden_layer(self, old_size: int, new_size: int):
        """Resize hidden layer while preserving learned patterns"""
        # Resize input-to-hidden weights
        new_input_hidden = np.random.randn(self.weights_input_hidden.shape[0], new_size) * 0.2
        copy_size = min(old_size, new_size)
        new_input_hidden[:, :copy_size] = self.weights_input_hidden[:, :copy_size]
        self.weights_input_hidden = new_input_hidden
        
        # Resize hidden biases
        new_bias_hidden = np.random.randn(new_size) * 0.1
        new_bias_hidden[:copy_size] = self.bias_hidden[:copy_size]
        self.bias_hidden = new_bias_hidden
        
        # Resize hidden-to-output weights
        new_hidden_output = np.random.randn(new_size, self.config.output_size) * 0.2
        new_hidden_output[:copy_size, :] = self.weights_hidden_output[:copy_size, :]
        self.weights_hidden_output = new_hidden_output
        
        # Resize recurrent weights if present
        if self.has_recurrent:
            new_recurrent = np.random.randn(new_size, new_size) * 0.1
            if self.weights_recurrent is not None:
                new_recurrent[:copy_size, :copy_size] = self.weights_recurrent[:copy_size, :copy_size]
            self.weights_recurrent = new_recurrent
        
        # Resize hidden state
        new_hidden_state = np.zeros(new_size)
        new_hidden_state[:copy_size] = self.hidden_state[:copy_size]
        self.hidden_state = new_hidden_state
    
    def crossover(self, other: 'EvolutionaryNeuralNetwork') -> 'EvolutionaryNeuralNetwork':
        """Create offspring by combining two evolutionary networks"""
        offspring = EvolutionaryNeuralNetwork(self.config)
        
        # Inherit structure from one parent
        if random.random() < 0.5:
            offspring.hidden_size = self.hidden_size
            offspring.has_recurrent = self.has_recurrent
        else:
            offspring.hidden_size = other.hidden_size
            offspring.has_recurrent = other.has_recurrent
        
        # Initialize offspring with proper structure
        offspring._initialize_structure()
        
        # Crossover weights
        offspring._crossover_weights(self, other)
        
        # Crossover behavioral traits
        offspring.exploration_drive = (self.exploration_drive + other.exploration_drive) / 2
        offspring.social_weight = (self.social_weight + other.social_weight) / 2
        offspring.memory_decay = (self.memory_decay + other.memory_decay) / 2
        
        # Add small random variations
        offspring.exploration_drive += np.random.normal(0, 0.05)
        offspring.social_weight += np.random.normal(0, 0.05)
        offspring.memory_decay += np.random.normal(0, 0.02)
        
        # Clip values
        offspring.exploration_drive = np.clip(offspring.exploration_drive, 0, 1)
        offspring.social_weight = np.clip(offspring.social_weight, 0, 1)
        offspring.memory_decay = np.clip(offspring.memory_decay, 0.5, 1.0)
        
        return offspring
    
    def _initialize_structure(self):
        """Initialize network structure based on current configuration"""
        self.weights_input_hidden = np.random.randn(self.config.min_input_size + self.config.memory_size * 2, self.hidden_size) * 0.2
        self.bias_hidden = np.random.randn(self.hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.config.output_size) * 0.2
        self.bias_output = np.random.randn(self.config.output_size) * 0.1
        
        if self.has_recurrent:
            self.weights_recurrent = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        else:
            self.weights_recurrent = None
        
        self.hidden_state = np.zeros(self.hidden_size)
        self.memory.clear()
    
    def _crossover_weights(self, parent1: 'EvolutionaryNeuralNetwork', parent2: 'EvolutionaryNeuralNetwork'):
        """Crossover weight matrices from two parents"""
        # Determine the size to use (minimum of parents and offspring)
        min_hidden = min(self.hidden_size, parent1.hidden_size, parent2.hidden_size)
        min_input = min(self.weights_input_hidden.shape[0], 
                       parent1.weights_input_hidden.shape[0], 
                       parent2.weights_input_hidden.shape[0])
        
        # Crossover input-to-hidden weights
        for i in range(min_input):
            for j in range(min_hidden):
                if random.random() < 0.5:
                    self.weights_input_hidden[i, j] = parent1.weights_input_hidden[i, j]
                else:
                    self.weights_input_hidden[i, j] = parent2.weights_input_hidden[i, j]
        
        # Crossover biases
        for i in range(min_hidden):
            if random.random() < 0.5:
                self.bias_hidden[i] = parent1.bias_hidden[i]
            else:
                self.bias_hidden[i] = parent2.bias_hidden[i]
        
        # Crossover hidden-to-output weights
        for i in range(min_hidden):
            for j in range(self.config.output_size):
                if random.random() < 0.5:
                    self.weights_hidden_output[i, j] = parent1.weights_hidden_output[i, j]
                else:
                    self.weights_hidden_output[i, j] = parent2.weights_hidden_output[i, j]
        
        # Crossover output biases
        for i in range(self.config.output_size):
            if random.random() < 0.5:
                self.bias_output[i] = parent1.bias_output[i]
            else:
                self.bias_output[i] = parent2.bias_output[i]
        
        # Crossover recurrent weights if both parents have them
        if self.has_recurrent and parent1.has_recurrent and parent2.has_recurrent:
            for i in range(min_hidden):
                for j in range(min_hidden):
                    if random.random() < 0.5:
                        self.weights_recurrent[i, j] = parent1.weights_recurrent[i, j]
                    else:
                        self.weights_recurrent[i, j] = parent2.weights_recurrent[i, j]
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the network structure and traits"""
        return {
            'hidden_size': self.hidden_size,
            'has_recurrent': self.has_recurrent,
            'exploration_drive': self.exploration_drive,
            'social_weight': self.social_weight,
            'memory_decay': self.memory_decay,
            'memory_length': len(self.memory),
            'decisions_made': self.decisions_made,
            'fitness_score': self.fitness_score
        }
