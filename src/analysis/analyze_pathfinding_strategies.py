#!/usr/bin/env python3
"""
Analysis of Current Pathfinding Strategies
Examining how agents find food, avoid predators, and hunt prey
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.neural.neural_agents import NeuralAgent, NeuralEnvironment
from src.neural.neural_network import SensorSystem
from src.core.ecosystem import SpeciesType, Position
import numpy as np
import math

def analyze_pathfinding_strategies():
    """Analyze the current pathfinding and navigation strategies"""
    print("🧭 CURRENT PATHFINDING STRATEGIES ANALYSIS")
    print("=" * 80)
    
    # Create test environment
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=False)
    
    print("🔍 PATHFINDING SYSTEM OVERVIEW:")
    print("-" * 50)
    
    # Analyze sensory system
    print("\n📡 SENSORY INPUTS (Neural Network Inputs):")
    print("  [0] Energy level (0-1) - Internal state")
    print("  [1] Age (0-1) - Experience factor")
    print("  [2] Distance to nearest food (0-1) - Food pathfinding")
    print("  [3] Angle to nearest food (-1 to 1) - Food direction")
    print("  [4] Distance to nearest threat/prey (0-1) - Combat pathfinding")
    print("  [5] Angle to nearest threat/prey (-1 to 1) - Combat direction")
    print("  [6] Population density (0-1) - Crowding awareness")
    print("  [7] Reproduction readiness (0-1) - Behavioral state")
    print("  [8] X boundary distance (0-1) - Boundary avoidance")
    print("  [9] Y boundary distance (0-1) - Boundary avoidance")
    
    print("\n🎯 MOVEMENT OUTPUTS:")
    print("  [0] Move X (-1 to 1) - Horizontal movement")
    print("  [1] Move Y (-1 to 1) - Vertical movement")
    print("  [2] Reproduce (0/1) - Reproduction decision")
    print("  [3] Intensity (0-1) - Movement speed multiplier")
    
    # Test different scenarios
    test_scenarios = [
        ("Food Seeking", create_food_seeking_scenario),
        ("Predator Avoidance", create_predator_avoidance_scenario),
        ("Prey Hunting", create_prey_hunting_scenario),
        ("Boundary Navigation", create_boundary_scenario)
    ]
    
    for scenario_name, scenario_func in test_scenarios:
        print(f"\n" + "=" * 80)
        print(f"🎬 SCENARIO: {scenario_name}")
        print("-" * 50)
        
        agent, env_state = scenario_func(env)
        analyze_scenario_pathfinding(agent, env_state, scenario_name)

def create_food_seeking_scenario(env):
    """Create a food-seeking scenario"""
    # Create herbivore
    agent = NeuralAgent(SpeciesType.HERBIVORE, Position(30, 30), agent_id=1)
    
    # Add food at different distances
    from src.core.ecosystem import Food
    env.food_sources = [
        Food(Position(40, 35)),  # Close food
        Food(Position(70, 70)),  # Distant food
        Food(Position(20, 80))   # Alternative food
    ]
    
    return agent, env

def create_predator_avoidance_scenario(env):
    """Create a predator avoidance scenario"""
    # Create herbivore (prey)
    herbivore = NeuralAgent(SpeciesType.HERBIVORE, Position(50, 50), agent_id=1)
    
    # Add carnivore threat
    carnivore = NeuralAgent(SpeciesType.CARNIVORE, Position(60, 55), agent_id=2)
    env.agents = [herbivore, carnivore]
    
    # Add food so herbivore has competing objectives
    from src.core.ecosystem import Food
    env.food_sources = [Food(Position(65, 60))]  # Food near predator
    
    return herbivore, env

def create_prey_hunting_scenario(env):
    """Create a prey hunting scenario"""
    # Create carnivore (hunter)
    carnivore = NeuralAgent(SpeciesType.CARNIVORE, Position(40, 40), agent_id=1)
    
    # Add herbivore prey
    herbivore = NeuralAgent(SpeciesType.HERBIVORE, Position(60, 50), agent_id=2)
    env.agents = [carnivore, herbivore]
    
    return carnivore, env

def create_boundary_scenario(env):
    """Create a boundary navigation scenario"""
    # Create agent near boundary
    agent = NeuralAgent(SpeciesType.HERBIVORE, Position(5, 5), agent_id=1)
    
    # Add food in center to test boundary vs food priority
    from src.core.ecosystem import Food
    env.food_sources = [Food(Position(50, 50))]
    
    return agent, env

def analyze_scenario_pathfinding(agent, env, scenario_name):
    """Analyze pathfinding behavior in a specific scenario"""
    
    # Get sensory inputs
    sensory_inputs = SensorSystem.get_sensory_inputs(agent, env)
    
    # Get neural decision
    neural_outputs = agent.brain.forward(np.array(sensory_inputs))
    actions = SensorSystem.interpret_network_output(neural_outputs)
    
    print(f"📊 Agent Position: ({agent.position.x:.1f}, {agent.position.y:.1f})")
    print(f"👁️ Vision Range: {agent.vision_range} units")
    print(f"🏃 Speed: {agent.speed}")
    
    # Analyze key pathfinding inputs
    print("\n🧠 PATHFINDING INPUTS:")
    if scenario_name == "Food Seeking":
        food_distance = sensory_inputs[2]
        food_angle = sensory_inputs[3]
        print(f"  Food distance: {food_distance:.3f} (0=on food, 1=far/none)")
        print(f"  Food angle: {food_angle:.3f} (direction to food)")
        
    elif scenario_name == "Predator Avoidance":
        threat_distance = sensory_inputs[4]
        threat_angle = sensory_inputs[5]
        food_distance = sensory_inputs[2]
        print(f"  Threat distance: {threat_distance:.3f} (0=on threat, 1=far/none)")
        print(f"  Threat angle: {threat_angle:.3f} (direction to threat)")
        print(f"  Food distance: {food_distance:.3f} (competing objective)")
        
    elif scenario_name == "Prey Hunting":
        prey_distance = sensory_inputs[4]
        prey_angle = sensory_inputs[5]
        print(f"  Prey distance: {prey_distance:.3f} (0=on prey, 1=far/none)")
        print(f"  Prey angle: {prey_angle:.3f} (direction to prey)")
        
    elif scenario_name == "Boundary Navigation":
        x_boundary = sensory_inputs[8]
        y_boundary = sensory_inputs[9]
        food_distance = sensory_inputs[2]
        print(f"  X boundary distance: {x_boundary:.3f} (0=at boundary, 1=center)")
        print(f"  Y boundary distance: {y_boundary:.3f} (0=at boundary, 1=center)")
        print(f"  Food distance: {food_distance:.3f} (competing objective)")
    
    # Analyze movement decision
    print("\n🎯 PATHFINDING DECISION:")
    move_x = actions['move_x']
    move_y = actions['move_y']
    intensity = actions['intensity']
    
    movement_magnitude = math.sqrt(move_x**2 + move_y**2)
    if movement_magnitude > 0:
        move_angle = math.atan2(move_y, move_x) * 180 / math.pi
        print(f"  Movement direction: {move_angle:.1f}° ({interpret_direction(move_angle)})")
        print(f"  Movement magnitude: {movement_magnitude:.3f}")
        print(f"  Movement intensity: {intensity:.3f}")
        
        # Calculate actual movement distance
        actual_distance = agent.speed * intensity * 0.5
        print(f"  Actual movement distance: {actual_distance:.2f} units")
    else:
        print(f"  Decision: Stay still")
    
    # Strategy assessment
    print("\n🧭 PATHFINDING STRATEGY ASSESSMENT:")
    assess_pathfinding_strategy(agent, env, sensory_inputs, actions, scenario_name)

def interpret_direction(angle_degrees):
    """Convert angle to cardinal direction"""
    angle = angle_degrees % 360
    if -22.5 <= angle <= 22.5 or 337.5 <= angle <= 360:
        return "East"
    elif 22.5 < angle <= 67.5:
        return "Northeast"
    elif 67.5 < angle <= 112.5:
        return "North"
    elif 112.5 < angle <= 157.5:
        return "Northwest"
    elif 157.5 < angle <= 202.5:
        return "West"
    elif 202.5 < angle <= 247.5:
        return "Southwest"
    elif 247.5 < angle <= 292.5:
        return "South"
    elif 292.5 < angle <= 337.5:
        return "Southeast"
    else:
        return "Unknown"

def assess_pathfinding_strategy(agent, env, inputs, actions, scenario_name):
    """Assess the effectiveness of the pathfinding strategy"""
    
    if scenario_name == "Food Seeking":
        food_distance = inputs[2]
        food_angle = inputs[3]
        
        if food_distance < 0.1:  # Very close to food
            if abs(actions['move_x']) < 0.2 and abs(actions['move_y']) < 0.2:
                print("  ✅ GOOD: Staying still near food (optimal foraging)")
            else:
                print("  ⚠️ CONCERN: Moving away from nearby food")
        elif food_distance < 0.5:  # Moderate distance
            # Check if movement is toward food
            expected_x = 1 if food_angle > 0 else -1 if food_angle < -0.5 else 0
            expected_y = 1 if 0.5 > food_angle > -0.5 else -1 if food_angle < -0.5 or food_angle > 0.5 else 0
            
            if (actions['move_x'] * expected_x > 0) or (actions['move_y'] * expected_y > 0):
                print("  ✅ GOOD: Moving toward food source")
            else:
                print("  ❌ POOR: Moving away from food source")
        else:
            print("  🔍 EXPLORING: No food in range, movement patterns unclear")
    
    elif scenario_name == "Predator Avoidance":
        threat_distance = inputs[4]
        threat_angle = inputs[5]
        food_distance = inputs[2]
        
        if threat_distance < 0.3:  # Close threat
            # Should move away from threat
            expected_escape_angle = (threat_angle + 1.0) % 2.0 - 1.0  # Opposite direction
            if abs(actions['move_x']) > 0.3 or abs(actions['move_y']) > 0.3:
                print("  ✅ GOOD: High movement intensity to escape threat")
            else:
                print("  ❌ DANGEROUS: Not fleeing from nearby threat")
        else:
            print("  ℹ️ INFO: No immediate threat, normal foraging behavior")
    
    elif scenario_name == "Prey Hunting":
        prey_distance = inputs[4]
        prey_angle = inputs[5]
        
        if prey_distance < 0.5:  # Prey in range
            if abs(actions['move_x']) > 0.2 or abs(actions['move_y']) > 0.2:
                print("  ✅ GOOD: Actively pursuing prey")
            else:
                print("  ⚠️ CONCERN: Not pursuing nearby prey")
        else:
            print("  🔍 SEARCHING: No prey in range, exploring")
    
    elif scenario_name == "Boundary Navigation":
        x_boundary = inputs[8]
        y_boundary = inputs[9]
        food_distance = inputs[2]
        
        min_boundary = min(x_boundary, y_boundary)
        if min_boundary < 0.2:  # Near boundary
            if food_distance < 0.3:  # Food nearby
                print("  🤔 COMPLEX: Balancing boundary avoidance with food acquisition")
            else:
                # Should move toward center
                center_direction_needed = x_boundary < 0.5 or y_boundary < 0.5
                if center_direction_needed and (abs(actions['move_x']) > 0.1 or abs(actions['move_y']) > 0.1):
                    print("  ✅ GOOD: Moving away from boundary")
                else:
                    print("  ⚠️ CONCERN: Not avoiding boundary effectively")
        else:
            print("  ✅ SAFE: Good distance from boundaries")

def main():
    print("🎯 PATHFINDING STRATEGIES SUMMARY")
    print("=" * 80)
    
    analyze_pathfinding_strategies()
    
    print("\n" + "=" * 80)
    print("📋 CURRENT PATHFINDING ARCHITECTURE:")
    print("1. 🧠 NEURAL NETWORK BASED: Decisions made through 10→12→4 neural network")
    print("2. 👁️ VISION-LIMITED: Agents only sense within vision range (15-20 units)")
    print("3. 🎯 SINGLE TARGET: Finds nearest food/threat/prey, no multi-target planning")
    print("4. 📐 ANGLE + DISTANCE: Uses polar coordinates for navigation")
    print("5. 🚫 NO PATHFINDING: Direct movement, no obstacle avoidance")
    print("6. 🧬 EVOLUTIONARY: Strategies evolve through genetic algorithm")
    print("7. ⚡ REACTIVE: Immediate response to current sensory input")
    print("8. 🎛️ INTENSITY CONTROL: Variable movement speed based on neural output")
    
    print("\n🧬 EVOLUTIONARY ENHANCEMENT OPPORTUNITIES:")
    print("=" * 50)
    print("To make this truly evolution-driven, agents need enhanced capabilities:")
    print("1. 🧠 MEMORY SYSTEM: Short-term memory for recent locations/experiences")
    print("2. � MULTI-TARGET SENSING: See multiple food/threat sources simultaneously")
    print("3. 📡 COMMUNICATION: Simple signals between nearby agents")
    print("4. 🎯 VELOCITY AWARENESS: Sense movement direction of other agents")
    print("5. 🗺️ SPATIAL MEMORY: Remember productive areas over time")
    print("6. 🔄 RECURRENT CONNECTIONS: Enable temporal learning patterns")
    print("7. 🌐 VARIABLE NETWORK SIZE: Evolution can grow/shrink neural complexity")
    print("8. 🎲 EXPLORATION BIAS: Intrinsic motivation to explore unknown areas")
    
    print("\n🔧 CURRENT DESIGN-DRIVEN LIMITATIONS TO REMOVE:")
    print("❌ Fixed 'nearest only' target selection")
    print("❌ No memory of previous locations or food sources") 
    print("❌ No path planning around obstacles")
    print("❌ No cooperative or flocking behaviors")
    print("❌ No predictive movement (intercepting moving targets)")
    print("❌ No area coverage strategies for exploration")
    print("❌ Simple greedy decision making")
    
    print("\n🎯 PROPOSED EVOLUTIONARY ARCHITECTURE:")
    print("✅ 15-20 input neurons with rich environmental data")
    print("✅ Variable hidden layer size (evolved trait)")
    print("✅ 3-cell memory buffer for recent states")
    print("✅ Multi-target sensing (3 nearest of each type)")
    print("✅ Communication output channel")
    print("✅ Recurrent connections for temporal patterns")
    print("✅ Exploration reward bonus in fitness function")
    print("✅ Social learning through observation")

if __name__ == "__main__":
    main()
