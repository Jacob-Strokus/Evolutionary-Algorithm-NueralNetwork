#!/usr/bin/env python3
"""
Boundary Clustering Investigation
Analyze why agents are moving toward and clustering at boundaries instead of eating food in the center
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.ecosystem import Environment, Position, SpeciesType
from src.neural.neural_agents import NeuralAgent, NeuralEnvironment
from src.neural.neural_network import NeuralNetworkConfig, SensorSystem
import numpy as np
import math

def create_test_environment():
    """Create a controlled test environment"""
    env = NeuralEnvironment(width=200, height=200, use_neural_agents=False)
    
    # Add some food sources in the center
    from src.core.ecosystem import Food
    
    # Center food
    env.food_sources.append(Food(Position(100, 100)))
    env.food_sources.append(Food(Position(90, 110)))
    env.food_sources.append(Food(Position(110, 90)))
    
    # Edge food for comparison
    env.food_sources.append(Food(Position(10, 10)))
    env.food_sources.append(Food(Position(190, 190)))
    
    return env

def analyze_agent_at_position(agent, env, position_name):
    """Analyze what an agent senses and decides at a specific position"""
    print(f"\n🔍 Analysis for agent at {position_name} (x={agent.position.x:.1f}, y={agent.position.y:.1f})")
    print("-" * 70)
    
    # Get sensory inputs
    sensory_inputs = SensorSystem.get_sensory_inputs(agent, env)
    
    print("📊 Sensory Inputs:")
    input_labels = [
        "Energy level", "Age", "Distance to food", "Angle to food",
        "Distance to threat/prey", "Angle to threat/prey", "Population density", 
        "Reproduction readiness", "X boundary distance", "Y boundary distance"
    ]
    
    for i, (label, value) in enumerate(zip(input_labels, sensory_inputs)):
        print(f"  [{i}] {label:20}: {value:.3f}")
    
    # Get neural network decision
    neural_outputs = agent.brain.forward(np.array(sensory_inputs))
    actions = SensorSystem.interpret_network_output(neural_outputs)
    
    print("\n🧠 Neural Network Outputs:")
    output_labels = ["Move X", "Move Y", "Reproduce", "Intensity"]
    for i, (label, value) in enumerate(zip(output_labels, neural_outputs)):
        print(f"  [{i}] {label:12}: {value:.3f}")
    
    print("\n🎯 Interpreted Actions:")
    for key, value in actions.items():
        print(f"  {key:15}: {value}")
    
    # Calculate movement direction
    move_magnitude = math.sqrt(actions['move_x']**2 + actions['move_y']**2)
    if move_magnitude > 0:
        move_angle = math.atan2(actions['move_y'], actions['move_x']) * 180 / math.pi
        print(f"  movement_angle : {move_angle:.1f}°")
        print(f"  movement_mag   : {move_magnitude:.3f}")
    
    # Check food proximity
    nearest_food = env.find_nearest_food(agent)
    if nearest_food:
        food_distance = agent.position.distance_to(nearest_food.position)
        food_dx = nearest_food.position.x - agent.position.x
        food_dy = nearest_food.position.y - agent.position.y
        food_angle = math.atan2(food_dy, food_dx) * 180 / math.pi
        print(f"\n🍎 Nearest Food:")
        print(f"  Distance: {food_distance:.2f}")
        print(f"  Direction: {food_angle:.1f}°")
        print(f"  Position: ({nearest_food.position.x:.1f}, {nearest_food.position.y:.1f})")
        
        # Check if movement is toward or away from food
        move_toward_food = (actions['move_x'] * food_dx + actions['move_y'] * food_dy) > 0
        print(f"  Moving toward food: {move_toward_food}")
    
    # Check boundary proximity
    left_dist = agent.position.x
    right_dist = env.width - agent.position.x
    top_dist = agent.position.y
    bottom_dist = env.height - agent.position.y
    
    print(f"\n🏁 Boundary Distances:")
    print(f"  Left: {left_dist:.1f}, Right: {right_dist:.1f}")
    print(f"  Top: {top_dist:.1f}, Bottom: {bottom_dist:.1f}")
    print(f"  Closest: {min(left_dist, right_dist, top_dist, bottom_dist):.1f}")
    
    return sensory_inputs, actions

def main():
    print("🔎 BOUNDARY CLUSTERING INVESTIGATION")
    print("=" * 80)
    print("Analyzing why agents move toward boundaries instead of center food...")
    
    # Create test environment
    env = create_test_environment()
    print(f"✅ Test environment created: {env.width}x{env.height}")
    print(f"🍎 Food sources: {len(env.food_sources)}")
    
    # Create test agents at different positions
    test_positions = [
        ("Corner (boundary)", Position(10, 10)),
        ("Edge (boundary)", Position(10, 100)),
        ("Center (safe)", Position(100, 100)),
        ("Near center", Position(80, 120)),
        ("Other corner", Position(190, 190))
    ]
    
    for position_name, position in test_positions:
        # Create a fresh agent at this position
        agent = NeuralAgent(SpeciesType.HERBIVORE, position, agent_id=1)
        agent.energy = 80  # Consistent energy level
        
        # Analyze this agent's behavior
        sensory_inputs, actions = analyze_agent_at_position(agent, env, position_name)
    
    print("\n" + "=" * 80)
    print("🎯 ANALYSIS SUMMARY:")
    print("Look for patterns in the outputs above:")
    print("1. Do boundary distance inputs (8,9) properly signal danger at boundaries?")
    print("2. Are agents moving toward or away from food sources?")
    print("3. What do the neural network weights favor - food or boundaries?")
    print("4. Are the boundary awareness inputs being interpreted correctly?")
    
    # Additional test: Create agent with specific boundary situation
    print("\n" + "=" * 80)
    print("🧪 BOUNDARY SENSITIVITY TEST")
    print("Testing how boundary inputs change with position...")
    
    test_agent = NeuralAgent(SpeciesType.HERBIVORE, Position(100, 100), agent_id=999)
    
    # Test positions from center to boundary
    test_x_positions = [100, 80, 60, 40, 20, 10, 5]  # Moving toward left boundary
    
    print("\nX-Position -> Boundary Input [8] (higher = safer):")
    for x_pos in test_x_positions:
        test_agent.position.x = x_pos
        test_agent.position.y = 100  # Keep Y constant
        
        sensory_inputs = SensorSystem.get_sensory_inputs(test_agent, env)
        boundary_x_input = sensory_inputs[8]
        
        distance_to_nearest_x = min(x_pos, env.width - x_pos)
        print(f"  X={x_pos:3.0f} -> boundary_input={boundary_x_input:.3f} (distance={distance_to_nearest_x:.1f})")

if __name__ == "__main__":
    main()
