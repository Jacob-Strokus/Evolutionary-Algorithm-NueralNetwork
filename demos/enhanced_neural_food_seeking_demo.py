#!/usr/bin/env python3
"""
Enhanced Neural Network Food-Seeking Demo - v3.0.3
Demonstrates improved neural network learning with unit vector directional inputs
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.ecosystem import Environment, Position, SpeciesType
from src.neural.neural_agents import NeuralAgent, NeuralEnvironment
from src.neural.neural_network import SensorSystem
import numpy as np
import random
import time

class EnhancedNeuralDemo:
    """Demo showcasing enhanced neural network food-seeking capabilities"""
    
    def __init__(self):
        self.environment = NeuralEnvironment(width=150, height=150, use_neural_agents=True)
        self.generation = 1
        print("🧠 Enhanced Neural Network Food-Seeking Demo")
        print("=" * 50)
        print(f"Environment: {self.environment.width}x{self.environment.height}")
        print(f"Initial population: {len(self.environment.agents)} neural agents")
    
    def analyze_neural_inputs(self, agent):
        """Analyze and display neural network sensory inputs"""
        try:
            inputs = SensorSystem.get_sensory_inputs(agent, self.environment)
            
            print(f"\n🧠 Neural Sensory Analysis (Agent {agent.id}):")
            print(f"  Energy Level: {inputs[0]:.3f} (0.0=empty, 1.0=full)")
            print(f"  Age Factor: {inputs[1]:.3f} (0.0=young, 1.0=old)")
            print(f"  Food Distance: {inputs[2]:.3f} (0.0=close, 1.0=far/none)")
            
            # Enhanced directional inputs
            if inputs[3] != 0 or inputs[4] != 0:
                magnitude = (inputs[3]**2 + inputs[4]**2)**0.5
                print(f"  Food Direction: X={inputs[3]:.3f}, Y={inputs[4]:.3f}")
                print(f"  Direction Vector Magnitude: {magnitude:.3f} (should be 1.0)")
                
                # Show which direction this points
                if abs(inputs[3]) > abs(inputs[4]):
                    direction = "East" if inputs[3] > 0 else "West"
                else:
                    direction = "North" if inputs[4] > 0 else "South"
                print(f"  Primary Direction: {direction}")
            else:
                print(f"  Food Direction: None visible")
            
            if len(inputs) > 5:
                print(f"  Threat/Prey Distance: {inputs[5]:.3f}")
                if inputs[6] != 0 or inputs[7] != 0:
                    print(f"  Threat/Prey Direction: X={inputs[6]:.3f}, Y={inputs[7]:.3f}")
            
            return inputs
        except Exception as e:
            print(f"  ❌ Error analyzing inputs: {e}")
            return None
    
    def test_movement_alignment(self, agent):
        """Test how well the agent's movement aligns with food direction"""
        try:
            inputs = SensorSystem.get_sensory_inputs(agent, self.environment)
            decision = agent.make_neural_decision(self.environment)
            
            print(f"\n🎯 Movement Analysis:")
            print(f"  Neural Decision: X={decision['move_x']:.3f}, Y={decision['move_y']:.3f}")
            print(f"  Movement Intensity: {decision['intensity']:.3f}")
            
            if inputs[3] != 0 or inputs[4] != 0:
                # Calculate alignment (dot product)
                alignment = inputs[3] * decision['move_x'] + inputs[4] * decision['move_y']
                alignment_percentage = alignment * 100
                
                print(f"  Food Direction: X={inputs[3]:.3f}, Y={inputs[4]:.3f}")
                print(f"  Movement Alignment: {alignment:.3f} ({alignment_percentage:+.1f}%)")
                
                if alignment > 0.5:
                    print(f"  Status: ✅ EXCELLENT - Moving toward food")
                elif alignment > 0:
                    print(f"  Status: ✅ GOOD - Generally toward food")
                elif alignment > -0.3:
                    print(f"  Status: ⚠️ NEUTRAL - Some food-seeking")
                else:
                    print(f"  Status: ❌ POOR - Moving away from food")
                
                return alignment
            else:
                print(f"  Status: 🔍 SEARCHING - No food visible")
                return 0
        except Exception as e:
            print(f"  ❌ Error in movement analysis: {e}")
            return 0
    
    def run_evolution_demonstration(self, generations=3, steps_per_generation=50):
        """Demonstrate evolution improving food-seeking behavior"""
        print(f"\n🧬 Evolution Demonstration ({generations} generations)")
        print("=" * 50)
        
        evolution_data = []
        
        for gen in range(generations):
            print(f"\n📊 Generation {gen + 1}")
            print("-" * 30)
            
            # Run simulation
            food_consumed = 0
            movement_alignments = []
            brain_fitness_scores = []
            
            for step in range(steps_per_generation):
                self.environment.step()
                
                # Track food consumption
                current_food = len([f for f in self.environment.food_sources if f.is_available])
                if step == 0:
                    initial_food = current_food
                
                if step == steps_per_generation - 1:
                    food_consumed = initial_food - current_food
            
            # Analyze generation results
            alive_agents = [a for a in self.environment.agents if a.is_alive]
            
            for agent in alive_agents[:5]:  # Test first 5 agents
                agent.update_fitness(self.environment)
                brain_fitness_scores.append(agent.brain.fitness_score)
                
                # Test movement alignment
                alignment = self.test_movement_alignment(agent)
                if alignment is not None:
                    movement_alignments.append(alignment)
            
            # Generation statistics
            gen_stats = {
                'generation': gen + 1,
                'population': len(alive_agents),
                'food_consumed': food_consumed,
                'avg_fitness': sum(brain_fitness_scores) / len(brain_fitness_scores) if brain_fitness_scores else 0,
                'max_fitness': max(brain_fitness_scores) if brain_fitness_scores else 0,
                'avg_alignment': sum(movement_alignments) / len(movement_alignments) if movement_alignments else 0,
                'positive_alignments': sum(1 for a in movement_alignments if a > 0),
                'total_tested': len(movement_alignments)
            }
            
            evolution_data.append(gen_stats)
            
            print(f"Population: {gen_stats['population']} agents")
            print(f"Food consumed: {gen_stats['food_consumed']} sources")
            print(f"Brain fitness: avg={gen_stats['avg_fitness']:.1f}, max={gen_stats['max_fitness']:.1f}")
            print(f"Movement: {gen_stats['positive_alignments']}/{gen_stats['total_tested']} agents toward food")
            print(f"Avg alignment: {gen_stats['avg_alignment']:.3f}")
            
            # Evolve to next generation
            if gen < generations - 1:
                self.environment.trigger_evolutionary_events()
                print("🔄 Evolved to next generation")
        
        # Show evolution summary
        self.show_evolution_summary(evolution_data)
        return evolution_data
    
    def show_evolution_summary(self, evolution_data):
        """Display comprehensive evolution summary"""
        print(f"\n📈 EVOLUTION SUMMARY")
        print("=" * 50)
        
        if len(evolution_data) >= 2:
            first_gen = evolution_data[0]
            last_gen = evolution_data[-1]
            
            # Calculate improvements
            fitness_improvement = ((last_gen['avg_fitness'] / max(first_gen['avg_fitness'], 1)) - 1) * 100
            food_improvement = last_gen['food_consumed'] - first_gen['food_consumed']
            alignment_improvement = last_gen['avg_alignment'] - first_gen['avg_alignment']
            
            print(f"Generations analyzed: {len(evolution_data)}")
            print(f"Population: {first_gen['population']} → {last_gen['population']} agents")
            print(f"Food consumption: {first_gen['food_consumed']} → {last_gen['food_consumed']} (+{food_improvement})")
            print(f"Brain fitness: {first_gen['avg_fitness']:.1f} → {last_gen['avg_fitness']:.1f} ({fitness_improvement:+.1f}%)")
            print(f"Movement alignment: {first_gen['avg_alignment']:.3f} → {last_gen['avg_alignment']:.3f} ({alignment_improvement:+.3f})")
            
            # Overall assessment
            improvements = sum([
                fitness_improvement > 5,
                food_improvement > 0,
                alignment_improvement > 0.1,
                last_gen['positive_alignments'] > first_gen['positive_alignments']
            ])
            
            if improvements >= 3:
                print("\n✅ EXCELLENT EVOLUTION: Neural networks learning effectively!")
            elif improvements >= 2:
                print("\n✅ GOOD EVOLUTION: Clear learning progress detected")
            elif improvements >= 1:
                print("\n⚠️ MODERATE EVOLUTION: Some improvement, needs more time")
            else:
                print("\n❌ POOR EVOLUTION: Learning may need adjustment")
    
    def demonstrate_individual_agent(self):
        """Detailed analysis of a single agent's behavior"""
        print(f"\n🔬 Individual Agent Analysis")
        print("=" * 50)
        
        alive_agents = [a for a in self.environment.agents if a.is_alive]
        if not alive_agents:
            print("❌ No alive agents to analyze")
            return
        
        agent = alive_agents[0]
        print(f"Analyzing Agent {agent.id} ({agent.species_type.value})")
        print(f"Position: ({agent.position.x:.1f}, {agent.position.y:.1f})")
        print(f"Energy: {agent.energy:.1f}/{agent.max_energy}")
        
        # Analyze sensory inputs
        inputs = self.analyze_neural_inputs(agent)
        
        # Test movement alignment
        alignment = self.test_movement_alignment(agent)
        
        # Show nearby food
        food_sources = [f for f in self.environment.food_sources if f.is_available]
        if food_sources:
            distances = [agent.position.distance_to(f.position) for f in food_sources]
            nearest_distance = min(distances)
            print(f"\n🍃 Food Environment:")
            print(f"  Available food sources: {len(food_sources)}")
            print(f"  Nearest food distance: {nearest_distance:.1f}")
            print(f"  Vision range: {agent.vision_range}")
            print(f"  Food in range: {'Yes' if nearest_distance <= agent.vision_range else 'No'}")

def main():
    """Run the enhanced neural network food-seeking demonstration"""
    demo = EnhancedNeuralDemo()
    
    # Individual agent analysis
    demo.demonstrate_individual_agent()
    
    # Evolution demonstration
    evolution_results = demo.run_evolution_demonstration(
        generations=3,
        steps_per_generation=60
    )
    
    print(f"\n🎉 Enhanced Neural Network Demo Complete!")
    print("Key Features Demonstrated:")
    print("• Unit vector directional inputs for precise food location")
    print("• Movement alignment analysis for food-seeking behavior")
    print("• Evolutionary improvement in neural network performance")
    print("• Mathematical safety systems preventing infinite values")
    print("• Real-time fitness calculation and generation progression")
    
    return demo, evolution_results

if __name__ == "__main__":
    demo, results = main()
