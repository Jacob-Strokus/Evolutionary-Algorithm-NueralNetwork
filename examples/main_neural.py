"""
Phase 2 Main Runner - Neural Network Decision-Making
Choose between traditional rule-based AI and neural network AI
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.ecosystem import Environment
from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
import time

def run_neural_simulation(steps: int = 1000, display_interval: int = 50):
    """Run the neural network-powered ecosystem simulation"""
    print("🧠 Starting Neural Network AI Ecosystem - Phase 2")
    print("=" * 65)
    
    # Create neural environment
    env = NeuralEnvironment(width=200, height=200)
    
    # Display initial state
    stats = env.get_neural_stats()
    print(f"Initial Neural Population:")
    print(f"  🦌 Herbivores: {stats['herbivore_count']}")
    print(f"  🐺 Carnivores: {stats['carnivore_count']}")
    print(f"  🧠 Neural Agents: {stats['neural_agents']}")
    print(f"  🌾 Available Food: {stats['available_food']}")
    print()
    
    # Run neural simulation
    for step in range(steps):
        env.step()
        
        # Display stats periodically
        if step % display_interval == 0 or step == steps - 1:
            stats = env.get_neural_stats()
            print(f"Step {stats['time_step']:4d} | "
                  f"H: {stats['herbivore_count']:2d} (energy: {stats['avg_herbivore_energy']:5.1f}) | "
                  f"C: {stats['carnivore_count']:2d} (energy: {stats['avg_carnivore_energy']:5.1f}) | "
                  f"Fitness: {stats.get('avg_neural_fitness', 0):6.1f} | "
                  f"Food: {stats['available_food']:2d}")
            
            # Check for extinction
            if stats['total_population'] == 0:
                print("\n💀 All species have gone extinct!")
                break
            elif stats['herbivore_count'] == 0:
                print("\n🦌 Herbivores extinct! Neural carnivores learning to starve...")
            elif stats['carnivore_count'] == 0:
                print("\n🐺 Carnivores extinct! Neural herbivores adapting...")
        
        # Small delay for watchability
        time.sleep(0.005)
    
    # Final neural statistics
    print("\n" + "=" * 65)
    print("🏁 Neural Simulation Complete!")
    final_stats = env.get_neural_stats()
    print(f"Final Population after {final_stats['time_step']} steps:")
    print(f"  🦌 Herbivores: {final_stats['herbivore_count']}")
    print(f"  🐺 Carnivores: {final_stats['carnivore_count']}")
    print(f"  👥 Total Survived: {final_stats['total_population']}")
    print(f"  🧠 Average Neural Fitness: {final_stats.get('avg_neural_fitness', 0):.2f}")
    print(f"  🎯 Average Decisions Made: {final_stats.get('avg_decisions_made', 0):.1f}")
    print(f"  👶 Total Offspring: {final_stats.get('total_offspring_produced', 0)}")
    print(f"  🌾 Available Food: {final_stats['available_food']}")
    
    return env

def run_comparison_simulation(steps: int = 500):
    """Run both traditional and neural simulations for comparison"""
    print("⚖️  Comparison: Traditional AI vs Neural Network AI")
    print("=" * 70)
    
    print("\n🤖 Running Traditional Rule-Based AI...")
    traditional_env = Environment()
    for step in range(steps):
        traditional_env.step()
        if step % 100 == 0:
            stats = traditional_env.get_population_stats()
            print(f"Traditional Step {step}: H={stats['herbivore_count']}, C={stats['carnivore_count']}")
    
    trad_final = traditional_env.get_population_stats()
    
    print(f"\n🧠 Running Neural Network AI...")
    neural_env = NeuralEnvironment()
    for step in range(steps):
        neural_env.step()
        if step % 100 == 0:
            stats = neural_env.get_neural_stats()
            print(f"Neural Step {step}: H={stats['herbivore_count']}, C={stats['carnivore_count']}, "
                  f"Fitness={stats.get('avg_neural_fitness', 0):.1f}")
    
    neural_final = neural_env.get_neural_stats()
    
    # Comparison results
    print("\n" + "=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'Traditional':<15} {'Neural':<15} {'Winner'}")
    print("-" * 70)
    
    # Compare survival
    trad_survival = trad_final['total_population']
    neural_survival = neural_final['total_population']
    survival_winner = "Neural" if neural_survival > trad_survival else "Traditional" if trad_survival > neural_survival else "Tie"
    print(f"{'Final Population':<25} {trad_survival:<15} {neural_survival:<15} {survival_winner}")
    
    # Compare energy efficiency
    trad_energy = (trad_final['avg_herbivore_energy'] + trad_final['avg_carnivore_energy']) / 2
    neural_energy = (neural_final['avg_herbivore_energy'] + neural_final['avg_carnivore_energy']) / 2
    energy_winner = "Neural" if neural_energy > trad_energy else "Traditional" if trad_energy > neural_energy else "Tie"
    print(f"{'Average Energy':<25} {trad_energy:<15.1f} {neural_energy:<15.1f} {energy_winner}")
    
    # Neural-specific metrics
    print(f"{'Neural Fitness':<25} {'N/A':<15} {neural_final.get('avg_neural_fitness', 0):<15.1f} {'Neural'}")
    print(f"{'Decisions Made':<25} {'N/A':<15} {neural_final.get('avg_decisions_made', 0):<15.1f} {'Neural'}")
    
    return traditional_env, neural_env

def run_neural_learning_demo(steps: int = 300):
    """Demonstrate neural network learning over time"""
    print("🎓 Neural Learning Demo - Watch AI Improve Over Time")
    print("=" * 60)
    
    env = NeuralEnvironment()
    
    fitness_history = []
    
    for step in range(steps):
        env.step()
        
        if step % 30 == 0:
            stats = env.get_neural_stats()
            current_fitness = stats.get('avg_neural_fitness', 0)
            fitness_history.append(current_fitness)
            
            # Show learning progress
            if len(fitness_history) > 1:
                improvement = current_fitness - fitness_history[-2]
                trend = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                print(f"Step {step:3d}: Fitness {current_fitness:6.1f} {trend} "
                      f"(Change: {improvement:+5.1f}) | "
                      f"Pop: H={stats['herbivore_count']:2d} C={stats['carnivore_count']:2d}")
            else:
                print(f"Step {step:3d}: Fitness {current_fitness:6.1f} | "
                      f"Pop: H={stats['herbivore_count']:2d} C={stats['carnivore_count']:2d}")
    
    # Learning analysis
    print("\n" + "=" * 60)
    print("🧠 LEARNING ANALYSIS")
    print("=" * 60)
    
    if len(fitness_history) >= 3:
        initial_fitness = fitness_history[0]
        final_fitness = fitness_history[-1]
        max_fitness = max(fitness_history)
        
        total_improvement = final_fitness - initial_fitness
        learning_rate = total_improvement / len(fitness_history)
        
        print(f"Initial Fitness:  {initial_fitness:8.2f}")
        print(f"Final Fitness:    {final_fitness:8.2f}")
        print(f"Peak Fitness:     {max_fitness:8.2f}")
        print(f"Total Learning:   {total_improvement:+8.2f}")
        print(f"Learning Rate:    {learning_rate:+8.3f} per update")
        
        if total_improvement > 5:
            print("\n🎉 Significant learning detected! Neural networks are adapting!")
        elif total_improvement > 0:
            print("\n✅ Modest learning observed. Networks are slowly improving.")
        else:
            print("\n⚠️  Limited learning. May need parameter tuning.")
    
    return env, fitness_history

if __name__ == "__main__":
    print("🧠 Phase 2: Neural Network Decision-Making")
    print("=" * 50)
    
    choice = input("""
Choose simulation type:
1. 🧠 Neural Network Simulation (500 steps)
2. ⚖️  Traditional vs Neural Comparison
3. 🎓 Neural Learning Demo
4. 🚀 Extended Neural Simulation (1500 steps)

Enter choice (1-4): """).strip()
    
    if choice == "1":
        run_neural_simulation(steps=500, display_interval=50)
    elif choice == "2":
        run_comparison_simulation(steps=400)
    elif choice == "3":
        run_neural_learning_demo(steps=300)
    elif choice == "4":
        run_neural_simulation(steps=1500, display_interval=75)
    else:
        print("Running default neural simulation...")
        run_neural_simulation(steps=500, display_interval=50)
