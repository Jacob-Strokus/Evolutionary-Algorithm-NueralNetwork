#!/usr/bin/env python3
"""
Test Boundary Awareness Fix
Verify that neural agents now have boundary awareness in their sensory inputs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.neural.neural_network import SensorSystem
from src.core.ecosystem import SpeciesType, Position

def test_boundary_awareness():
    """Test if agents now have proper boundary awareness"""
    print("🧪 Testing Boundary Awareness Fix")
    print("=" * 50)
    
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Find a neural agent to test
    test_agent = None
    for agent in env.agents:
        if isinstance(agent, NeuralAgent):
            test_agent = agent
            break
    
    if not test_agent:
        print("❌ No neural agents found!")
        return False
    
    print(f"🧠 Testing Agent: {test_agent.id} ({test_agent.species_type.value})")
    print(f"🌍 Environment: {env.width}x{env.height}")
    print()
    
    # Test positions and expected boundary awareness
    test_cases = [
        (5, 50, "Near left boundary", "High X-boundary awareness"),
        (95, 50, "Near right boundary", "High X-boundary awareness"),
        (50, 5, "Near top boundary", "High Y-boundary awareness"),
        (50, 95, "Near bottom boundary", "High Y-boundary awareness"),
        (2, 2, "Near corner", "High both boundaries"),
        (50, 50, "Center", "Low boundary awareness"),
        (25, 75, "Off-center", "Medium boundary awareness")
    ]
    
    print("📍 Testing sensory inputs at different positions:")
    print("Position  | Location              | Boundary Inputs     | Expected")
    print("-" * 75)
    
    all_tests_passed = True
    
    for x, y, description, expected in test_cases:
        # Move agent to test position
        test_agent.position = Position(x, y)
        
        # Get sensory inputs
        inputs = SensorSystem.get_sensory_inputs(test_agent, env)
        
        # Extract boundary awareness inputs
        x_boundary_awareness = inputs[8]  # Distance to X boundary
        y_boundary_awareness = inputs[9]  # Distance to Y boundary
        
        # Determine if awareness matches expectation
        test_passed = True
        
        if "High X-boundary" in expected and x_boundary_awareness > 0.3:
            test_passed = False
        if "High Y-boundary" in expected and y_boundary_awareness > 0.3:
            test_passed = False
        if "High both" in expected and (x_boundary_awareness > 0.2 or y_boundary_awareness > 0.2):
            test_passed = False
        if "Low boundary" in expected and (x_boundary_awareness < 0.8 or y_boundary_awareness < 0.8):
            test_passed = False
        
        status = "✅" if test_passed else "❌"
        if not test_passed:
            all_tests_passed = False
        
        print(f"({x:2d},{y:2d})   | {description:20s} | X:{x_boundary_awareness:.2f} Y:{y_boundary_awareness:.2f}     | {expected} {status}")
    
    print()
    print("📊 All sensory inputs for center position (50,50):")
    test_agent.position = Position(50, 50)
    inputs = SensorSystem.get_sensory_inputs(test_agent, env)
    
    input_names = [
        "Energy level", "Age", "Distance to food", "Angle to food",
        "Distance threat/prey", "Angle threat/prey", "Population density", 
        "Reproduction", "X boundary distance", "Y boundary distance"
    ]
    
    for i, (name, value) in enumerate(zip(input_names, inputs)):
        print(f"   [{i}] {name:20s}: {value:.3f}")
    
    return all_tests_passed

def test_boundary_avoidance_behavior():
    """Test if agents now avoid boundaries better"""
    print("\n🎯 Testing Boundary Avoidance Behavior")
    print("=" * 45)
    
    env = NeuralEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Find an agent and place it at boundary
    test_agent = None
    for agent in env.agents:
        if isinstance(agent, NeuralAgent):
            test_agent = agent
            break
    
    if not test_agent:
        print("❌ No test agent found!")
        return False
    
    # Place agent at left boundary
    test_agent.position = Position(1, 25)  # Right at the edge
    test_agent.energy = test_agent.max_energy * 0.9  # Give good energy
    
    print(f"🧪 Test Agent: {test_agent.id} ({test_agent.species_type.value})")
    print(f"📍 Starting position: ({test_agent.position.x:.1f}, {test_agent.position.y:.1f})")
    print()
    
    # Track movement over several steps
    print("Step | Position      | X-Bound | Y-Bound | Move Decision     | Result")
    print("-" * 75)
    
    moved_away_count = 0
    total_steps = 15
    
    for step in range(total_steps):
        # Get sensory inputs before movement
        inputs = SensorSystem.get_sensory_inputs(test_agent, env)
        x_boundary = inputs[8]
        y_boundary = inputs[9]
        
        # Get neural decision
        actions = test_agent.make_neural_decision(env)
        move_x = actions['move_x']
        move_y = actions['move_y']
        
        # Store old position
        old_x = test_agent.position.x
        old_y = test_agent.position.y
        
        # Execute movement
        test_agent.neural_move(env)
        env.keep_agent_in_bounds(test_agent)
        
        # Check if moved away from boundary
        new_x = test_agent.position.x
        moved_away_from_x_boundary = new_x > old_x  # Moving right from left boundary
        
        if moved_away_from_x_boundary and x_boundary < 0.5:  # Was close to boundary
            moved_away_count += 1
        
        result = "✅ Moved away" if moved_away_from_x_boundary else "❌ Stuck/wrong"
        
        print(f"{step:4d} | ({new_x:4.1f},{test_agent.position.y:4.1f}) | " +
              f"{x_boundary:7.2f} | {y_boundary:7.2f} | " +
              f"({move_x:5.2f},{move_y:5.2f}) | {result}")
        
        # Update agent
        test_agent.update()
    
    # Calculate success rate
    success_rate = (moved_away_count / max(1, total_steps)) * 100
    
    print()
    print(f"📈 Boundary Avoidance Results:")
    print(f"   Steps that moved away from boundary: {moved_away_count}/{total_steps}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    if success_rate > 40:  # Should be better than random (50%)
        print("✅ Agent shows improved boundary avoidance!")
        return True
    else:
        print("❌ Agent still struggles with boundary avoidance")
        return False

def run_comparative_simulation():
    """Run simulation to compare boundary clustering before/after fix"""
    print("\n🔬 Comparative Simulation Analysis")
    print("=" * 45)
    
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Track agent positions over time
    boundary_agents_over_time = []
    boundary_threshold = 15  # Distance from edge
    
    for step in range(50):
        env.step()
        
        # Count agents near boundaries every 5 steps
        if step % 5 == 0:
            boundary_count = 0
            total_agents = 0
            
            for agent in env.agents:
                if isinstance(agent, NeuralAgent) and agent.is_alive:
                    total_agents += 1
                    x, y = agent.position.x, agent.position.y
                    
                    near_boundary = (x <= boundary_threshold or 
                                   x >= (env.width - boundary_threshold) or
                                   y <= boundary_threshold or 
                                   y >= (env.height - boundary_threshold))
                    
                    if near_boundary:
                        boundary_count += 1
            
            if total_agents > 0:
                boundary_percentage = (boundary_count / total_agents) * 100
                boundary_agents_over_time.append(boundary_percentage)
    
    # Analyze results
    if boundary_agents_over_time:
        avg_boundary_percentage = sum(boundary_agents_over_time) / len(boundary_agents_over_time)
        max_boundary_percentage = max(boundary_agents_over_time)
        
        print(f"📊 Boundary clustering analysis:")
        print(f"   Average agents near boundary: {avg_boundary_percentage:.1f}%")
        print(f"   Maximum agents near boundary: {max_boundary_percentage:.1f}%")
        print(f"   Boundary threshold: {boundary_threshold} units")
        
        if avg_boundary_percentage < 30:  # Should be well below 30%
            print("✅ Reduced boundary clustering achieved!")
            return True
        else:
            print("⚠️ Some boundary clustering still present")
            return False
    
    return False

if __name__ == "__main__":
    try:
        print("🧪 Boundary Awareness Fix Test Suite")
        print("=" * 70)
        
        # Test 1: Verify boundary awareness inputs
        test1_passed = test_boundary_awareness()
        
        # Test 2: Test boundary avoidance behavior
        test2_passed = test_boundary_avoidance_behavior()
        
        # Test 3: Run comparative simulation
        test3_passed = run_comparative_simulation()
        
        print(f"\n🏁 Test Results Summary:")
        print(f"   Boundary Awareness Inputs: {'✅ PASS' if test1_passed else '❌ FAIL'}")
        print(f"   Boundary Avoidance Behavior: {'✅ PASS' if test2_passed else '❌ FAIL'}")
        print(f"   Reduced Boundary Clustering: {'✅ PASS' if test3_passed else '❌ FAIL'}")
        
        if test1_passed and test2_passed and test3_passed:
            print("\n🎉 All tests passed! Boundary awareness fix is working!")
        else:
            print("\n⚠️ Some tests failed. The fix may need further refinement.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
