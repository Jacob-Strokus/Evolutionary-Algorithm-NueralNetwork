"""
Phase 2: Neural Network Decision-Making System
Replaces simple rule-based AI with trainable neural networks
"""
import numpy as np
import random
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class NeuralNetworkConfig:
    """Configuration for neural network architecture"""
    input_size: int = 10  # Number of sensory inputs (increased for boundary awareness)
    hidden_size: int = 12  # Hidden layer neurons
    output_size: int = 4  # Number of actions
    learning_rate: float = 0.01
    mutation_rate: float = 0.1
    mutation_strength: float = 0.2

class NeuralNetwork:
    """Simple feedforward neural network for agent decision-making"""
    
    def __init__(self, config: NeuralNetworkConfig):
        self.config = config
        
        # Initialize weights with small random values
        self.weights_input_hidden = np.random.randn(config.input_size, config.hidden_size) * 0.5
        self.bias_hidden = np.random.randn(config.hidden_size) * 0.1
        
        self.weights_hidden_output = np.random.randn(config.hidden_size, config.output_size) * 0.5
        self.bias_output = np.random.randn(config.output_size) * 0.1
        
        # Track network performance
        self.fitness_score = 0.0
        self.age = 0
        self.decisions_made = 0
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(np.clip(x, -500, 500))
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        # Ensure inputs are numpy array
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        
        # Handle backward compatibility for networks with 8 inputs when 10 are provided
        if inputs.shape[0] == 10 and self.weights_input_hidden.shape[0] == 8:
            # Use only the first 8 inputs for old networks
            inputs = inputs[:8]
        elif inputs.shape[0] == 8 and self.weights_input_hidden.shape[0] == 10:
            # Pad with zeros for boundary awareness if old inputs provided to new network
            boundary_inputs = np.array([0.5, 0.5])  # Default to "center" boundary awareness
            inputs = np.concatenate([inputs, boundary_inputs])
        
        # Input to hidden layer
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.tanh(hidden_input)
        
        # Hidden to output layer
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.sigmoid(output_input)
        
        self.decisions_made += 1
        return output
    
    def mutate(self):
        """Apply random mutations to the network weights"""
        mutation_rate = self.config.mutation_rate
        mutation_strength = self.config.mutation_strength
        
        # Mutate input-to-hidden weights
        mutation_mask = np.random.random(self.weights_input_hidden.shape) < mutation_rate
        mutations = np.random.randn(*self.weights_input_hidden.shape) * mutation_strength
        self.weights_input_hidden += mutation_mask * mutations
        
        # Mutate hidden biases
        mutation_mask = np.random.random(self.bias_hidden.shape) < mutation_rate
        mutations = np.random.randn(*self.bias_hidden.shape) * mutation_strength
        self.bias_hidden += mutation_mask * mutations
        
        # Mutate hidden-to-output weights
        mutation_mask = np.random.random(self.weights_hidden_output.shape) < mutation_rate
        mutations = np.random.randn(*self.weights_hidden_output.shape) * mutation_strength
        self.weights_hidden_output += mutation_mask * mutations
        
        # Mutate output biases
        mutation_mask = np.random.random(self.bias_output.shape) < mutation_rate
        mutations = np.random.randn(*self.bias_output.shape) * mutation_strength
        self.bias_output += mutation_mask * mutations
    
    def crossover(self, other: 'NeuralNetwork') -> 'NeuralNetwork':
        """Create offspring by combining two neural networks"""
        offspring = NeuralNetwork(self.config)
        
        # Blend weights using random interpolation
        alpha = np.random.random()
        
        offspring.weights_input_hidden = (alpha * self.weights_input_hidden + 
                                        (1 - alpha) * other.weights_input_hidden)
        offspring.bias_hidden = alpha * self.bias_hidden + (1 - alpha) * other.bias_hidden
        
        offspring.weights_hidden_output = (alpha * self.weights_hidden_output + 
                                         (1 - alpha) * other.weights_hidden_output)
        offspring.bias_output = alpha * self.bias_output + (1 - alpha) * other.bias_output
        
        return offspring
    
    def copy(self) -> 'NeuralNetwork':
        """Create a copy of this neural network"""
        copy_net = NeuralNetwork(self.config)
        copy_net.weights_input_hidden = self.weights_input_hidden.copy()
        copy_net.bias_hidden = self.bias_hidden.copy()
        copy_net.weights_hidden_output = self.weights_hidden_output.copy()
        copy_net.bias_output = self.bias_output.copy()
        return copy_net
    
    def upgrade_to_boundary_awareness(self):
        """Upgrade an 8-input network to support 10-input boundary awareness"""
        if self.weights_input_hidden.shape[0] == 8:
            # Create new weights with 2 additional inputs for boundary awareness
            old_weights = self.weights_input_hidden.copy()
            
            # Initialize new weights matrix (10 inputs instead of 8)
            new_weights = np.random.randn(10, self.config.hidden_size) * 0.1
            
            # Copy existing weights for the first 8 inputs
            new_weights[:8, :] = old_weights
            
            # Initialize boundary awareness weights (inputs 8 and 9) with small values
            # These should encourage movement away from boundaries
            new_weights[8:10, :] = np.random.randn(2, self.config.hidden_size) * 0.3
            
            self.weights_input_hidden = new_weights
            self.config.input_size = 10

class SensorSystem:
    """Handles sensory input processing for neural network agents"""
    
    @staticmethod
    def get_sensory_inputs(agent, environment) -> List[float]:
        """
        Extract sensory information for the neural network
        Returns 10 normalized inputs:
        [0] Energy level (0-1)
        [1] Age normalized (0-1)
        [2] Distance to nearest food (0-1)
        [3] Angle to nearest food (-1 to 1)
        [4] Distance to nearest threat/prey (0-1)
        [5] Angle to nearest threat/prey (-1 to 1)
        [6] Population density nearby (0-1)
        [7] Reproduction readiness (0-1)
        [8] Distance to nearest X boundary (0-1)
        [9] Distance to nearest Y boundary (0-1)
        """
        inputs = [0.0] * 10
        
        # Energy level (normalized to max energy)
        inputs[0] = min(1.0, agent.energy / agent.max_energy)
        
        # Age (normalized to expected lifespan)
        inputs[1] = min(1.0, agent.age / 1000.0)
        
        # Find nearest food
        nearest_food = environment.find_nearest_food(agent)
        if nearest_food:
            distance = agent.position.distance_to(nearest_food.position)
            inputs[2] = min(1.0, distance / agent.vision_range)
            
            # Angle to food
            dx = nearest_food.position.x - agent.position.x
            dy = nearest_food.position.y - agent.position.y
            angle = math.atan2(dy, dx) / math.pi  # Normalize to -1 to 1
            inputs[3] = angle
        else:
            inputs[2] = 1.0  # No food visible
            inputs[3] = 0.0
        
        # Find nearest threat or prey
        if agent.species_type.value == "herbivore":
            nearest_threat = environment.find_nearest_threat(agent)
            if nearest_threat:
                distance = agent.position.distance_to(nearest_threat.position)
                inputs[4] = min(1.0, distance / agent.vision_range)
                
                dx = nearest_threat.position.x - agent.position.x
                dy = nearest_threat.position.y - agent.position.y
                angle = math.atan2(dy, dx) / math.pi
                inputs[5] = angle
            else:
                inputs[4] = 1.0  # No threat visible
                inputs[5] = 0.0
        else:  # Carnivore
            nearest_prey = environment.find_nearest_prey(agent)
            if nearest_prey:
                distance = agent.position.distance_to(nearest_prey.position)
                inputs[4] = min(1.0, distance / agent.vision_range)
                
                dx = nearest_prey.position.x - agent.position.x
                dy = nearest_prey.position.y - agent.position.y
                angle = math.atan2(dy, dx) / math.pi
                inputs[5] = angle
            else:
                inputs[4] = 1.0  # No prey visible
                inputs[5] = 0.0
        
        # Population density (count nearby agents)
        nearby_count = 0
        for other_agent in environment.agents:
            if other_agent != agent and other_agent.is_alive:
                distance = agent.position.distance_to(other_agent.position)
                if distance <= agent.vision_range:
                    nearby_count += 1
        inputs[6] = min(1.0, nearby_count / 10.0)  # Normalize to max 10 nearby agents
        
        # Reproduction readiness
        inputs[7] = 1.0 if agent.can_reproduce() else 0.0
        
        # Distance to nearest X boundary (0 = at boundary, 1 = center)
        distance_to_left = agent.position.x
        distance_to_right = environment.width - agent.position.x
        min_x_distance = min(distance_to_left, distance_to_right)
        max_x_distance = environment.width / 2  # Maximum possible distance from center
        
        # FIXED: Invert boundary signal - higher values = safer (away from boundary)
        x_center_ratio = min_x_distance / max_x_distance
        inputs[8] = min(1.0, x_center_ratio)
        
        # Distance to nearest Y boundary (0 = at boundary, 1 = center)  
        distance_to_top = agent.position.y
        distance_to_bottom = environment.height - agent.position.y
        min_y_distance = min(distance_to_top, distance_to_bottom)
        max_y_distance = environment.height / 2  # Maximum possible distance from center
        
        # FIXED: Invert boundary signal - higher values = safer (away from boundary)
        y_center_ratio = min_y_distance / max_y_distance
        inputs[9] = min(1.0, y_center_ratio)
        
        return inputs
    
    @staticmethod
    def interpret_network_output(outputs: np.ndarray) -> dict:
        """
        Convert neural network outputs to agent actions
        4 outputs:
        [0] Move X direction (-1 to 1)
        [1] Move Y direction (-1 to 1) 
        [2] Try to reproduce (>0.5 = yes)
        [3] Action intensity/speed multiplier (0-1)
        """
        # Convert sigmoid outputs (0-1) to meaningful ranges
        move_x = (outputs[0] - 0.5) * 2  # Convert to -1 to 1
        move_y = (outputs[1] - 0.5) * 2  # Convert to -1 to 1
        should_reproduce = outputs[2] > 0.5
        intensity = outputs[3]  # Keep as 0-1
        
        return {
            'move_x': move_x,
            'move_y': move_y,
            'should_reproduce': should_reproduce,
            'intensity': intensity
        }

def add_neural_networks_to_agent_class():
    """
    This function documents how to integrate neural networks into the existing Agent class.
    We'll create a new NeuralAgent class that inherits from Agent.
    """
    pass

if __name__ == "__main__":
    # Test neural network functionality
    print("🧠 Testing Neural Network System")
    print("=" * 50)
    
    # Create a neural network
    config = NeuralNetworkConfig()
    brain = NeuralNetwork(config)
    
    # Test with random inputs
    test_inputs = np.random.random(10)  # Updated for 10 inputs
    outputs = brain.forward(test_inputs)
    actions = SensorSystem.interpret_network_output(outputs)
    
    print(f"✅ Neural network created: {config.input_size} → {config.hidden_size} → {config.output_size}")
    print(f"📊 Test inputs: {[f'{x:.2f}' for x in test_inputs]}")
    print(f"🎮 Network outputs: {[f'{x:.2f}' for x in outputs]}")
    print(f"🎯 Interpreted actions: {actions}")
    
    # Test mutation
    original_weights = brain.weights_input_hidden.copy()
    brain.mutate()
    changes = np.sum(np.abs(brain.weights_input_hidden - original_weights))
    print(f"🧬 Mutation applied, total weight changes: {changes:.3f}")
    
    # Test crossover
    brain2 = NeuralNetwork(config)
    offspring = brain.crossover(brain2)
    print(f"👶 Offspring created through crossover")
    
    print("\n🚀 Neural network system ready for integration!")
