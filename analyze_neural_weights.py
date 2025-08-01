#!/usr/bin/env python3
"""
Neural Network Weight Analysis
Investigate why neural networks are not learning to approach food
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.neural.neural_agents import NeuralAgent
from src.neural.neural_network import NeuralNetworkConfig, SensorSystem
from src.core.ecosystem import Position, SpeciesType
import numpy as np

def analyze_neural_weights():
    """Analyze the neural network weights to understand decision bias"""
    print("🧠 NEURAL NETWORK WEIGHT ANALYSIS")
    print("=" * 80)
    
    # Create a fresh neural agent
    agent = NeuralAgent(SpeciesType.HERBIVORE, Position(100, 100), agent_id=1)
    brain = agent.brain
    
    print(f"Network architecture: {brain.config.input_size} → {brain.config.hidden_size} → {brain.config.output_size}")
    print(f"Fitness score: {brain.fitness_score}")
    print(f"Decisions made: {brain.decisions_made}")
    
    # Analyze input-to-hidden weights
    print("\n📊 INPUT-TO-HIDDEN WEIGHTS:")
    print("Input connections that most influence hidden neurons...")
    
    input_labels = [
        "Energy", "Age", "Food Dist", "Food Angle", 
        "Threat Dist", "Threat Angle", "Population", "Reproduction",
        "X Boundary", "Y Boundary"
    ]
    
    # Find which inputs have the strongest weights
    weights_ih = brain.weights_input_hidden
    print(f"Weights shape: {weights_ih.shape}")
    
    # Calculate average absolute weight for each input
    avg_weights_per_input = np.mean(np.abs(weights_ih), axis=1)
    
    print("\nAverage absolute weight per input (higher = more influential):")
    for i, (label, avg_weight) in enumerate(zip(input_labels, avg_weights_per_input)):
        print(f"  [{i:2}] {label:12}: {avg_weight:.3f}")
    
    # Find the most influential inputs
    most_influential = np.argsort(avg_weights_per_input)[::-1]
    print(f"\nMost influential inputs (ranked):")
    for rank, input_idx in enumerate(most_influential[:5]):
        print(f"  {rank+1}. {input_labels[input_idx]} (weight: {avg_weights_per_input[input_idx]:.3f})")
    
    # Analyze hidden-to-output weights
    print("\n📈 HIDDEN-TO-OUTPUT WEIGHTS:")
    weights_ho = brain.weights_hidden_output
    output_labels = ["Move X", "Move Y", "Reproduce", "Intensity"]
    
    print(f"Weights shape: {weights_ho.shape}")
    print("Average weights for each output:")
    
    for i, label in enumerate(output_labels):
        avg_weight = np.mean(weights_ho[:, i])
        std_weight = np.std(weights_ho[:, i])
        print(f"  {label:10}: avg={avg_weight:+.3f}, std={std_weight:.3f}")
    
    # Test specific scenarios
    print("\n🧪 SCENARIO TESTING:")
    print("Testing neural responses to specific input patterns...")
    
    # Scenario 1: Food directly ahead (0 distance, 0 angle)
    test_inputs_food_ahead = [0.5, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    outputs_food_ahead = brain.forward(np.array(test_inputs_food_ahead))
    actions_food_ahead = SensorSystem.interpret_network_output(outputs_food_ahead)
    
    print("\nScenario 1: Food directly ahead (distance=0, angle=0)")
    print(f"  Inputs: {test_inputs_food_ahead}")
    print(f"  Outputs: {[f'{x:.3f}' for x in outputs_food_ahead]}")
    print(f"  Actions: move_x={actions_food_ahead['move_x']:+.3f}, move_y={actions_food_ahead['move_y']:+.3f}")
    print(f"  Result: {'Moving toward food' if abs(actions_food_ahead['move_x']) < 0.1 and abs(actions_food_ahead['move_y']) < 0.1 else 'Moving away from food'}")
    
    # Scenario 2: Food to the right (distance=0.5, angle=0)
    test_inputs_food_right = [0.5, 0.1, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    outputs_food_right = brain.forward(np.array(test_inputs_food_right))
    actions_food_right = SensorSystem.interpret_network_output(outputs_food_right)
    
    print("\nScenario 2: Food to the right (distance=0.5, angle=0)")
    print(f"  Inputs: {test_inputs_food_right}")
    print(f"  Outputs: {[f'{x:.3f}' for x in outputs_food_right]}")
    print(f"  Actions: move_x={actions_food_right['move_x']:+.3f}, move_y={actions_food_right['move_y']:+.3f}")
    print(f"  Expected: move_x > 0 (move right toward food)")
    print(f"  Actual: {'✅ Correct' if actions_food_right['move_x'] > 0.1 else '❌ Wrong direction'}")
    
    # Scenario 3: At boundary with food in center
    test_inputs_boundary = [0.5, 0.1, 0.8, 0.5, 1.0, 0.0, 0.0, 0.0, 0.1, 0.1]  # Near boundary
    outputs_boundary = brain.forward(np.array(test_inputs_boundary))
    actions_boundary = SensorSystem.interpret_network_output(outputs_boundary)
    
    print("\nScenario 3: At boundary with food in center")
    print(f"  Inputs: {test_inputs_boundary}")
    print(f"  Outputs: {[f'{x:.3f}' for x in outputs_boundary]}")
    print(f"  Actions: move_x={actions_boundary['move_x']:+.3f}, move_y={actions_boundary['move_y']:+.3f}")
    print(f"  Expected: Move toward center (away from boundary)")
    
    return brain

def test_gradient_response():
    """Test how network responds to gradual changes in food distance"""
    print("\n" + "=" * 80)
    print("🎯 GRADIENT RESPONSE TEST")
    print("Testing how movement changes as food gets closer...")
    
    agent = NeuralAgent(SpeciesType.HERBIVORE, Position(100, 100), agent_id=1)
    brain = agent.brain
    
    food_distances = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]  # Far to very close
    
    print("\nFood Distance -> Movement Response:")
    print("Distance | Move X | Move Y | Expected")
    print("-" * 40)
    
    for distance in food_distances:
        # Food directly to the right (angle = 0)
        test_inputs = [0.5, 0.1, distance, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        outputs = brain.forward(np.array(test_inputs))
        actions = SensorSystem.interpret_network_output(outputs)
        
        expected = "Move right" if distance > 0 else "Stay/eat"
        actual = "Right" if actions['move_x'] > 0.1 else "Left" if actions['move_x'] < -0.1 else "Stay"
        status = "✅" if (distance > 0 and actions['move_x'] > 0.1) or (distance == 0 and abs(actions['move_x']) < 0.2) else "❌"
        
        print(f"  {distance:.1f}   | {actions['move_x']:+.3f} | {actions['move_y']:+.3f} | {expected:8} {status}")

def main():
    brain = analyze_neural_weights()
    test_gradient_response()
    
    print("\n" + "=" * 80)
    print("🔍 DIAGNOSIS SUMMARY:")
    print("1. Check if food distance/angle inputs have low influence weights")
    print("2. Check if boundary inputs have disproportionately high influence")
    print("3. Look for bias in output weights that prevent food-seeking behavior")
    print("4. The network might need retraining or better initialization")

if __name__ == "__main__":
    main()
