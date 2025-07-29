#!/usr/bin/env python3
"""
Boundary Clustering Investigation
Investigate why agents cluster at environment boundaries
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType, Position
import matplotlib.pyplot as plt
import numpy as np

def analyze_boundary_clustering():
    """Analyze agent positions over time to detect boundary clustering"""
    print("🔍 Investigating Boundary Clustering")
    print("=" * 60)
    
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Track agent positions over time
    position_history = {
        'herbivores': {'x': [], 'y': []},
        'carnivores': {'x': [], 'y': []}
    }
    
    boundary_threshold = 10  # Distance from edge to consider "boundary"
    
    print(f"🌍 Environment: {env.width}x{env.height}")
    print(f"🎯 Boundary threshold: {boundary_threshold} units from edge")
    print(f"🦌 Initial Herbivores: {len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE])}")
    print(f"🐺 Initial Carnivores: {len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.CARNIVORE])}")
    print()
    
    # Run simulation and track positions
    for step in range(100):
        env.step()
        
        # Record positions every 10 steps
        if step % 10 == 0:
            for agent in env.agents:
                if isinstance(agent, NeuralAgent) and agent.is_alive:
                    if agent.species_type == SpeciesType.HERBIVORE:
                        position_history['herbivores']['x'].append(agent.position.x)
                        position_history['herbivores']['y'].append(agent.position.y)
                    else:
                        position_history['carnivores']['x'].append(agent.position.x)
                        position_history['carnivores']['y'].append(agent.position.y)
    
    # Analyze boundary clustering
    def count_boundary_agents(positions_x, positions_y, width, height, threshold):
        """Count how many agents are near boundaries"""
        boundary_count = 0
        total_count = len(positions_x)
        
        for x, y in zip(positions_x, positions_y):
            near_left = x <= threshold
            near_right = x >= (width - threshold)
            near_top = y <= threshold
            near_bottom = y >= (height - threshold)
            
            if near_left or near_right or near_top or near_bottom:
                boundary_count += 1
        
        return boundary_count, total_count
    
    print("📊 Boundary Clustering Analysis:")
    print("-" * 40)
    
    # Analyze herbivores
    herb_boundary, herb_total = count_boundary_agents(
        position_history['herbivores']['x'],
        position_history['herbivores']['y'],
        env.width, env.height, boundary_threshold
    )
    herb_boundary_percent = (herb_boundary / max(1, herb_total)) * 100
    
    # Analyze carnivores
    carn_boundary, carn_total = count_boundary_agents(
        position_history['carnivores']['x'],
        position_history['carnivores']['y'],
        env.width, env.height, boundary_threshold
    )
    carn_boundary_percent = (carn_boundary / max(1, carn_total)) * 100
    
    print(f"🦌 Herbivores:")
    print(f"   Total positions sampled: {herb_total}")
    print(f"   Near boundary: {herb_boundary} ({herb_boundary_percent:.1f}%)")
    
    print(f"🐺 Carnivores:")
    print(f"   Total positions sampled: {carn_total}")
    print(f"   Near boundary: {carn_boundary} ({carn_boundary_percent:.1f}%)")
    print()
    
    # Check if clustering is significant (>25% at boundaries is concerning)
    if herb_boundary_percent > 25 or carn_boundary_percent > 25:
        print("❌ BOUNDARY CLUSTERING DETECTED!")
        print("   Agents are spending too much time near boundaries")
        
        # Test boundary awareness
        test_boundary_awareness(env)
        
        return False
    else:
        print("✅ No significant boundary clustering detected")
        return True

def test_boundary_awareness(env):
    """Test if agents have boundary awareness in their sensory inputs"""
    print("\n🧠 Testing Neural Network Boundary Awareness")
    print("=" * 50)
    
    # Find a neural agent
    neural_agent = None
    for agent in env.agents:
        if isinstance(agent, NeuralAgent):
            neural_agent = agent
            break
    
    if not neural_agent:
        print("❌ No neural agents found!")
        return
    
    # Place agent at different positions and check sensory inputs
    test_positions = [
        (5, 50, "Near left boundary"),      # Near left edge
        (95, 50, "Near right boundary"),    # Near right edge  
        (50, 5, "Near top boundary"),       # Near top edge
        (50, 95, "Near bottom boundary"),   # Near bottom edge
        (50, 50, "Center of environment")   # Center
    ]
    
    print("📍 Testing sensory inputs at different positions:")
    print("Pos  | Location              | Sensory Inputs (8 values)")
    print("-" * 65)
    
    from src.neural.neural_network import SensorSystem
    
    for x, y, description in test_positions:
        # Move agent to test position
        neural_agent.position = Position(x, y)
        
        # Get sensory inputs
        inputs = SensorSystem.get_sensory_inputs(neural_agent, env)
        
        # Format inputs for display
        input_str = " ".join([f"{inp:.2f}" for inp in inputs])
        
        print(f"({x:2d},{y:2d}) | {description:20s} | {input_str}")
    
    print("\n📝 Sensory Input Legend:")
    print("   [0] Energy level     [1] Age            [2] Distance to food")
    print("   [3] Angle to food    [4] Distance prey  [5] Angle to prey")
    print("   [6] Population dens. [7] Reproduction")
    print()
    print("❌ DIAGNOSIS: Neural networks have NO boundary awareness!")
    print("   Agents don't know how close they are to environment edges.")
    print("   When they move to boundaries, they get 'stuck' by position clamping.")

def demonstrate_boundary_problem():
    """Demonstrate the boundary problem with a specific example"""
    print("\n🎯 Demonstrating Boundary Problem")
    print("=" * 45)
    
    env = NeuralEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Find an agent and place it near boundary
    test_agent = None
    for agent in env.agents:
        if isinstance(agent, NeuralAgent):
            test_agent = agent
            break
    
    if not test_agent:
        print("❌ No test agent found!")
        return
    
    # Place agent near left boundary
    test_agent.position = Position(2, 25)  # Very close to left edge
    test_agent.energy = test_agent.max_energy * 0.8  # Give it good energy
    
    print(f"🧪 Test Agent ID: {test_agent.id} ({test_agent.species_type.value})")
    print(f"📍 Initial Position: ({test_agent.position.x:.1f}, {test_agent.position.y:.1f})")
    print(f"⚡ Energy: {test_agent.energy:.1f}")
    print()
    
    # Run simulation and track movement
    print("Step | Position     | Neural Outputs        | Bounded Pos   | Notes")
    print("-" * 75)
    
    for step in range(10):
        # Get neural decision before movement
        actions = test_agent.make_neural_decision(env)
        move_x = actions['move_x']
        move_y = actions['move_y']
        
        # Store position before bounds checking
        old_pos = (test_agent.position.x, test_agent.position.y)
        
        # Execute neural movement (this will try to move)
        test_agent.neural_move(env)
        
        # Check if position was clamped by boundaries
        env.keep_agent_in_bounds(test_agent)
        new_pos = (test_agent.position.x, test_agent.position.y)
        
        # Check if movement was blocked
        was_blocked = (old_pos[0] == new_pos[0] and old_pos[1] == new_pos[1] and 
                      (move_x != 0 or move_y != 0))
        
        notes = ""
        if was_blocked:
            notes += "🚫 BLOCKED "
        if new_pos[0] <= 1 or new_pos[0] >= env.width-1:
            notes += "📍 AT X-BOUNDARY "
        if new_pos[1] <= 1 or new_pos[1] >= env.height-1:
            notes += "📍 AT Y-BOUNDARY "
        
        print(f"{step:4d} | ({new_pos[0]:4.1f},{new_pos[1]:4.1f}) | " +
              f"({move_x:5.2f},{move_y:5.2f}) | " +
              f"({new_pos[0]:4.1f},{new_pos[1]:4.1f}) | {notes}")
        
        # Update agent for next step
        test_agent.update()
    
    final_x, final_y = test_agent.position.x, test_agent.position.y
    is_at_boundary = (final_x <= 5 or final_x >= env.width-5 or 
                     final_y <= 5 or final_y >= env.height-5)
    
    print()
    if is_at_boundary:
        print("❌ CONFIRMED: Agent ended up near boundary and may be stuck!")
        print("   Neural network keeps trying to move but boundary clamping prevents it.")
    else:
        print("✅ Agent successfully moved away from boundary")

if __name__ == "__main__":
    try:
        print("🧪 Boundary Clustering Investigation Suite")
        print("=" * 70)
        
        # Run main analysis
        clustering_detected = not analyze_boundary_clustering()
        
        if clustering_detected:
            # Demonstrate the specific problem
            demonstrate_boundary_problem()
            
            print(f"\n🎯 SOLUTION NEEDED:")
            print("   Add boundary awareness to neural network sensory inputs")
            print("   Add distance-to-boundary as inputs [8] and [9] (X and Y distances)")
            print("   This will allow agents to 'feel' the walls and move away from them")
        
        print(f"\n🏁 Investigation Complete")
        
    except Exception as e:
        print(f"❌ Investigation failed with error: {e}")
        import traceback
        traceback.print_exc()
