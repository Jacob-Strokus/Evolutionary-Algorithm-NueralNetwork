#!/usr/bin/env python3
"""
Verify Boundary Clustering Fix
Quick test to check if agents are staying near center food instead of clustering at boundaries
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.neural.neural_agents import NeuralEnvironment
from src.core.ecosystem import SpeciesType
import statistics

def test_agent_distribution():
    """Test if agents are distributed more toward center vs boundaries"""
    print("🔍 TESTING AGENT DISTRIBUTION AFTER BOUNDARY CLUSTERING FIX")
    print("=" * 80)
    
    # Create environment and run for a few steps
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    print(f"✅ Environment created: {len(env.agents)} agents")
    
    # Run simulation for 50 steps to let agents move
    for step in range(50):
        env.step()
        if step % 10 == 0:
            print(f"Step {step}: {len([a for a in env.agents if a.is_alive])} alive agents")
    
    # Analyze agent positions
    alive_agents = [a for a in env.agents if a.is_alive]
    center_x, center_y = env.width / 2, env.height / 2
    
    distances_from_center = []
    distances_from_boundary = []
    
    for agent in alive_agents:
        # Distance from center
        center_distance = ((agent.position.x - center_x)**2 + (agent.position.y - center_y)**2)**0.5
        distances_from_center.append(center_distance)
        
        # Distance from nearest boundary
        boundary_distance = min(
            agent.position.x,  # Left boundary
            env.width - agent.position.x,  # Right boundary
            agent.position.y,  # Top boundary
            env.height - agent.position.y  # Bottom boundary
        )
        distances_from_boundary.append(boundary_distance)
    
    if distances_from_center:
        avg_center_distance = statistics.mean(distances_from_center)
        avg_boundary_distance = statistics.mean(distances_from_boundary)
        
        print(f"\n📊 AGENT DISTRIBUTION ANALYSIS:")
        print(f"Total alive agents: {len(alive_agents)}")
        print(f"Average distance from center: {avg_center_distance:.2f}")
        print(f"Average distance from boundary: {avg_boundary_distance:.2f}")
        print(f"Max possible center distance: {(center_x**2 + center_y**2)**0.5:.2f}")
        print(f"Max possible boundary distance: {min(center_x, center_y):.2f}")
        
        # Calculate ratios
        center_ratio = avg_center_distance / ((center_x**2 + center_y**2)**0.5)
        boundary_ratio = avg_boundary_distance / min(center_x, center_y)
        
        print(f"\n🎯 CLUSTERING ASSESSMENT:")
        print(f"Center distance ratio: {center_ratio:.3f} (0.0=center, 1.0=corner)")
        print(f"Boundary distance ratio: {boundary_ratio:.3f} (0.0=boundary, 1.0=center)")
        
        if boundary_ratio > 0.4:
            print("✅ SUCCESS: Agents are staying away from boundaries!")
        elif boundary_ratio > 0.2:
            print("⚠️  PARTIAL: Some improvement in boundary avoidance")
        else:
            print("❌ ISSUE: Agents still clustering near boundaries")
        
        # Count agents in different zones
        center_zone = sum(1 for d in distances_from_center if d <= 15)  # Within 15 units of center
        boundary_zone = sum(1 for d in distances_from_boundary if d <= 10)  # Within 10 units of boundary
        
        print(f"\n🏃 AGENT ZONES:")
        print(f"Agents in center zone (≤15 from center): {center_zone} ({center_zone/len(alive_agents)*100:.1f}%)")
        print(f"Agents in boundary zone (≤10 from boundary): {boundary_zone} ({boundary_zone/len(alive_agents)*100:.1f}%)")
        
        # Food consumption check
        total_food_consumed = sum(getattr(agent, 'lifetime_food_consumed', 0) for agent in alive_agents)
        print(f"\n🍎 FOOD CONSUMPTION:")
        print(f"Total food consumed by all agents: {total_food_consumed}")
        print(f"Average food per agent: {total_food_consumed/len(alive_agents):.2f}")

if __name__ == "__main__":
    test_agent_distribution()
