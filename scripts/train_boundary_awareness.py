#!/usr/bin/env python3
"""
Boundary Awareness Training
Train agents to avoid boundaries using fitness-based learning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType, Position
import random

def train_boundary_awareness():
    """Train agents to avoid boundaries through fitness penalties"""
    print("🎓 Training Boundary Awareness")
    print("=" * 50)
    
    env = NeuralEnvironment(width=80, height=80, use_neural_agents=True)
    
    print(f"🌍 Training Environment: {env.width}x{env.height}")
    print(f"🦌 Herbivores: {len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE])}")
    print(f"🐺 Carnivores: {len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.CARNIVORE])}")
    print()
    
    # Track boundary behavior over training
    generation_data = []
    boundary_penalty_weight = 15.0  # How much to penalize boundary proximity
    
    print("Training Progress (50 steps per report):")
    print("Step | Boundary Avg | Best Fitness | Alive | Notes")
    print("-" * 60)
    
    for step in range(250):
        env.step()
        
        # Apply boundary fitness penalties during training
        for agent in env.agents:
            if isinstance(agent, NeuralAgent) and agent.is_alive:
                # Calculate boundary penalty
                x_boundary_distance = min(agent.position.x, env.width - agent.position.x)
                y_boundary_distance = min(agent.position.y, env.height - agent.position.y)
                min_boundary_distance = min(x_boundary_distance, y_boundary_distance)
                
                # Apply fitness penalty for being too close to boundaries
                if min_boundary_distance < 10:  # Within 10 units of boundary
                    penalty = (10 - min_boundary_distance) * boundary_penalty_weight
                    agent.brain.fitness_score -= penalty
                
                # Small bonus for staying in center area
                if min_boundary_distance > 20:
                    agent.brain.fitness_score += 2
        
        # Report progress every 50 steps
        if step % 50 == 0:
            alive_agents = [a for a in env.agents if isinstance(a, NeuralAgent) and a.is_alive]
            
            if alive_agents:
                # Calculate average boundary distance
                boundary_distances = []
                fitnesses = []
                
                for agent in alive_agents:
                    x_dist = min(agent.position.x, env.width - agent.position.x)
                    y_dist = min(agent.position.y, env.height - agent.position.y)
                    boundary_distances.append(min(x_dist, y_dist))
                    fitnesses.append(agent.brain.fitness_score)
                
                avg_boundary_dist = sum(boundary_distances) / len(boundary_distances)
                best_fitness = max(fitnesses)
                
                notes = ""
                if avg_boundary_dist > 15:
                    notes = "✅ Good boundary avoidance"
                elif avg_boundary_dist < 8:
                    notes = "❌ Clustering at boundaries"
                else:
                    notes = "⚠️ Learning in progress"
                
                print(f"{step:4d} | {avg_boundary_dist:11.1f} | {best_fitness:11.1f} | {len(alive_agents):5d} | {notes}")
                
                generation_data.append({
                    'step': step,
                    'avg_boundary_distance': avg_boundary_dist,
                    'best_fitness': best_fitness,
                    'alive_count': len(alive_agents)
                })
    
    # Final analysis
    print("\n📊 Training Results Analysis:")
    print("-" * 40)
    
    if generation_data:
        initial_boundary_dist = generation_data[0]['avg_boundary_distance']
        final_boundary_dist = generation_data[-1]['avg_boundary_distance']
        improvement = final_boundary_dist - initial_boundary_dist
        
        print(f"Initial average boundary distance: {initial_boundary_dist:.1f}")
        print(f"Final average boundary distance: {final_boundary_dist:.1f}")
        print(f"Improvement: {improvement:+.1f} units")
        
        if improvement > 3:
            print("✅ Significant improvement in boundary avoidance!")
            return True
        elif improvement > 0:
            print("⚠️ Some improvement, but could be better")
            return False
        else:
            print("❌ No improvement or regression in boundary behavior")
            return False
    
    return False

def test_trained_agents():
    """Test if trained agents show better boundary behavior"""
    print("\n🧪 Testing Trained Agent Boundary Behavior")
    print("=" * 50)
    
    env = NeuralEnvironment(width=60, height=60, use_neural_agents=True)
    
    # Run training first
    for _ in range(100):  # Quick training
        env.step()
        
        # Apply boundary penalties
        for agent in env.agents:
            if isinstance(agent, NeuralAgent) and agent.is_alive:
                x_dist = min(agent.position.x, env.width - agent.position.x)
                y_dist = min(agent.position.y, env.height - agent.position.y)
                min_dist = min(x_dist, y_dist)
                
                if min_dist < 8:
                    agent.brain.fitness_score -= (8 - min_dist) * 10
    
    # Now test a specific agent
    test_agents = [a for a in env.agents if isinstance(a, NeuralAgent) and a.is_alive]
    if not test_agents:
        print("❌ No agents survived training!")
        return False
    
    # Select the best performing agent
    best_agent = max(test_agents, key=lambda a: a.brain.fitness_score)
    
    print(f"🧠 Testing Best Agent: {best_agent.id} (fitness: {best_agent.brain.fitness_score:.1f})")
    
    # Place agent near boundary and observe behavior
    best_agent.position = Position(3, 30)  # Near left boundary
    
    print(f"📍 Starting Position: ({best_agent.position.x:.1f}, {best_agent.position.y:.1f})")
    print()
    print("Step | Position      | Boundary Dist | Movement        | Status")
    print("-" * 65)
    
    successful_escapes = 0
    
    for step in range(20):
        old_x, old_y = best_agent.position.x, best_agent.position.y
        
        # Get boundary distance before movement
        x_dist = min(old_x, env.width - old_x)
        y_dist = min(old_y, env.height - old_y)
        boundary_dist = min(x_dist, y_dist)
        
        # Execute neural movement
        actions = best_agent.make_neural_decision(env)
        best_agent.neural_move(env)
        env.keep_agent_in_bounds(best_agent)
        
        # Calculate movement
        new_x, new_y = best_agent.position.x, best_agent.position.y
        move_x = new_x - old_x
        move_y = new_y - old_y
        
        # Check if moved away from boundary
        new_boundary_dist = min(new_x, env.width - new_x, new_y, env.height - new_y)
        escaped_boundary = boundary_dist < 10 and new_boundary_dist > boundary_dist
        
        if escaped_boundary:
            successful_escapes += 1
        
        status = "✅ Escaped" if escaped_boundary else ("🚫 Stuck" if boundary_dist < 5 else "🚶 Moving")
        
        print(f"{step:4d} | ({new_x:4.1f},{new_y:4.1f}) | {boundary_dist:11.1f} | " +
              f"({move_x:5.2f},{move_y:5.2f}) | {status}")
        
        best_agent.update()
    
    escape_rate = (successful_escapes / 20) * 100
    print(f"\n📈 Escape Success Rate: {escape_rate:.1f}%")
    
    if escape_rate > 25:
        print("✅ Agent shows learned boundary avoidance!")
        return True
    else:
        print("❌ Agent still needs more training")
        return False

if __name__ == "__main__":
    try:
        print("🎓 Boundary Awareness Training Suite")
        print("=" * 70)
        
        # Run training
        training_successful = train_boundary_awareness()
        
        # Test trained agents
        testing_successful = test_trained_agents()
        
        print(f"\n🏁 Training Results:")
        print(f"   Population Training: {'✅ SUCCESS' if training_successful else '❌ NEEDS MORE WORK'}")
        print(f"   Individual Testing: {'✅ SUCCESS' if testing_successful else '❌ NEEDS MORE WORK'}")
        
        if training_successful and testing_successful:
            print("\n🎉 Boundary awareness training was successful!")
            print("   Agents have learned to avoid clustering at boundaries.")
        else:
            print("\n💡 Recommendation: Run longer training sessions or increase boundary penalties")
            
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
