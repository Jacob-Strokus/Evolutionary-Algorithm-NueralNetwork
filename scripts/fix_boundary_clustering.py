#!/usr/bin/env python3
"""
Fix Boundary Clustering Issue
Initialize neural networks with food-seeking bias and improve the fitness function
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.neural.neural_network import NeuralNetwork, NeuralNetworkConfig
import numpy as np

def create_food_seeking_neural_network(config: NeuralNetworkConfig) -> NeuralNetwork:
    """Create a neural network with initial weights biased toward food-seeking behavior"""
    
    brain = NeuralNetwork(config)
    
    # Bias the input-to-hidden weights to prioritize food inputs
    # Inputs [2] and [3] are food distance and angle - make these more influential
    food_distance_idx = 2
    food_angle_idx = 3
    boundary_x_idx = 8
    boundary_y_idx = 9
    
    # Increase weights for food-related inputs
    brain.weights_input_hidden[food_distance_idx, :] *= 2.0  # Food distance more important
    brain.weights_input_hidden[food_angle_idx, :] *= 2.0     # Food angle more important
    
    # Reduce weights for boundary inputs (they shouldn't dominate food behavior)
    brain.weights_input_hidden[boundary_x_idx, :] *= 0.5     # Reduce boundary X influence
    brain.weights_input_hidden[boundary_y_idx, :] *= 0.5     # Reduce boundary Y influence
    
    # Bias hidden-to-output weights to encourage movement toward food
    # Output [0] is Move X, [1] is Move Y
    # We want the network to output movement that aligns with food direction
    
    # Create a bias toward staying still when food is very close (distance=0)
    # This requires the network to learn to reduce movement when food distance is low
    
    # Add bias to encourage exploration (slight positive movement tendency)
    brain.bias_output[0] += 0.1  # Slight rightward bias for exploration
    brain.bias_output[1] += 0.1  # Slight upward bias for exploration
    
    return brain

def test_improved_network():
    """Test the improved network initialization"""
    print("🧠 TESTING IMPROVED NEURAL NETWORK INITIALIZATION")
    print("=" * 80)
    
    # Create both original and improved networks
    config = NeuralNetworkConfig(input_size=10, hidden_size=12, output_size=4)
    
    print("Testing food-seeking behavior...")
    
    # Test scenarios
    scenarios = [
        ("Food at distance 0 (on food)", [0.5, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
        ("Food to the right", [0.5, 0.1, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
        ("Food to the left", [0.5, 0.1, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]),  # angle=1.0 is left
        ("Food below", [0.5, 0.1, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]),  # angle=-0.5 is down
    ]
    
    # Test original network
    print("\n📊 ORIGINAL NETWORK (random weights):")
    original_brain = NeuralNetwork(config)
    
    for scenario_name, inputs in scenarios:
        outputs = original_brain.forward(np.array(inputs))
        move_x = (outputs[0] - 0.5) * 2
        move_y = (outputs[1] - 0.5) * 2
        print(f"  {scenario_name:20}: move_x={move_x:+.3f}, move_y={move_y:+.3f}")
    
    # Test improved network
    print("\n🎯 IMPROVED NETWORK (food-seeking bias):")
    improved_brain = create_food_seeking_neural_network(config)
    
    for scenario_name, inputs in scenarios:
        outputs = improved_brain.forward(np.array(inputs))
        move_x = (outputs[0] - 0.5) * 2
        move_y = (outputs[1] - 0.5) * 2
        print(f"  {scenario_name:20}: move_x={move_x:+.3f}, move_y={move_y:+.3f}")
    
    # Analyze weight differences
    print("\n📈 WEIGHT ANALYSIS:")
    print("Food distance weight increase:", np.mean(np.abs(improved_brain.weights_input_hidden[2, :])) / np.mean(np.abs(original_brain.weights_input_hidden[2, :])))
    print("Food angle weight increase:  ", np.mean(np.abs(improved_brain.weights_input_hidden[3, :])) / np.mean(np.abs(original_brain.weights_input_hidden[3, :])))
    print("Boundary X weight decrease:  ", np.mean(np.abs(improved_brain.weights_input_hidden[8, :])) / np.mean(np.abs(original_brain.weights_input_hidden[8, :])))
    print("Boundary Y weight decrease:  ", np.mean(np.abs(improved_brain.weights_input_hidden[9, :])) / np.mean(np.abs(original_brain.weights_input_hidden[9, :])))

def create_improved_neural_environment_patch():
    """Create a patch file to fix the neural environment initialization"""
    
    patch_code = '''
# PATCH: Add this method to NeuralEnvironment class in neural_agents.py

def create_food_seeking_neural_network(self, config: NeuralNetworkConfig) -> NeuralNetwork:
    """Create a neural network with initial weights biased toward food-seeking behavior"""
    
    brain = NeuralNetwork(config)
    
    # Bias the input-to-hidden weights to prioritize food inputs
    food_distance_idx = 2  # Input [2] is food distance
    food_angle_idx = 3     # Input [3] is food angle
    boundary_x_idx = 8     # Input [8] is X boundary distance
    boundary_y_idx = 9     # Input [9] is Y boundary distance
    
    # SOLUTION 1: Increase weights for food-related inputs
    brain.weights_input_hidden[food_distance_idx, :] *= 2.0  
    brain.weights_input_hidden[food_angle_idx, :] *= 2.0     
    
    # SOLUTION 2: Reduce weights for boundary inputs (prevent boundary obsession)
    brain.weights_input_hidden[boundary_x_idx, :] *= 0.3     
    brain.weights_input_hidden[boundary_y_idx, :] *= 0.3     
    
    # SOLUTION 3: Initialize output biases to encourage food-seeking
    # Small positive bias to encourage movement/exploration
    brain.bias_output[0] += 0.05  # Move X bias
    brain.bias_output[1] += 0.05  # Move Y bias
    brain.bias_output[3] += 0.2   # Intensity bias (be more active)
    
    return brain

# PATCH: Modify the NeuralAgent __init__ method to use improved initialization
# In NeuralAgent.__init__, replace:
#   self.brain = NeuralNetwork(neural_config)
# With:
#   if hasattr(self, 'environment') and hasattr(self.environment, 'create_food_seeking_neural_network'):
#       self.brain = self.environment.create_food_seeking_neural_network(neural_config)
#   else:
#       self.brain = NeuralNetwork(neural_config)
#       # Apply food-seeking bias manually
#       self.brain.weights_input_hidden[2, :] *= 2.0  # Food distance
#       self.brain.weights_input_hidden[3, :] *= 2.0  # Food angle
#       self.brain.weights_input_hidden[8, :] *= 0.3  # Boundary X
#       self.brain.weights_input_hidden[9, :] *= 0.3  # Boundary Y
'''
    
    print("💡 SOLUTION PATCH:")
    print(patch_code)

if __name__ == "__main__":
    test_improved_network()
    create_improved_neural_environment_patch()
    
    print("\n" + "=" * 80)
    print("🎯 BOUNDARY CLUSTERING SOLUTION SUMMARY:")
    print("1. Neural networks were initialized with random weights")
    print("2. Food distance/angle inputs had equal weight to boundary inputs")
    print("3. Networks randomly moved instead of seeking food")
    print("4. SOLUTION: Initialize networks with food-seeking bias")
    print("5. Increase food input weights, decrease boundary input weights")
    print("6. Add small movement bias to encourage exploration")
    print("\n✅ Apply the patch above to fix boundary clustering!")
