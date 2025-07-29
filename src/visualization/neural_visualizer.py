"""
Neural Network Ecosystem Visualizer for Phase 2
Enhanced visualization showing neural network decisions and fitness
"""
import matplotlib.pyplot as plt
import numpy as np
from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType

def visualize_neural_ecosystem(env: NeuralEnvironment, save_path: str = None):
    """Create visualization showing neural agents and their fitness"""
    # Set matplotlib to use non-interactive backend for WSL/headless environments
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Main ecosystem view
    ax1.set_title(f'Neural Ecosystem - Step {env.time_step}', fontsize=14, fontweight='bold')
    
    # Plot food sources
    food_x = [food.position.x for food in env.food_sources if food.is_available]
    food_y = [food.position.y for food in env.food_sources if food.is_available]
    ax1.scatter(food_x, food_y, c='brown', s=40, alpha=0.7, label='Food', marker='s')
    
    # Plot neural agents with fitness-based coloring
    neural_agents = [agent for agent in env.agents if isinstance(agent, NeuralAgent)]
    
    if neural_agents:
        herbivores = [agent for agent in neural_agents if agent.species_type == SpeciesType.HERBIVORE]
        carnivores = [agent for agent in neural_agents if agent.species_type == SpeciesType.CARNIVORE]
        
        if herbivores:
            herb_x = [agent.position.x for agent in herbivores]
            herb_y = [agent.position.y for agent in herbivores]
            herb_fitness = [agent.brain.fitness_score for agent in herbivores]
            
            scatter_h = ax1.scatter(herb_x, herb_y, c=herb_fitness, s=80, alpha=0.8, 
                                  label='Neural Herbivores', marker='o', cmap='Greens', 
                                  vmin=0, vmax=max(30, max(herb_fitness) if herb_fitness else 30))
            plt.colorbar(scatter_h, ax=ax1, label='Herbivore Fitness', shrink=0.6)
        
        if carnivores:
            carn_x = [agent.position.x for agent in carnivores]
            carn_y = [agent.position.y for agent in carnivores]
            carn_fitness = [agent.brain.fitness_score for agent in carnivores]
            
            scatter_c = ax1.scatter(carn_x, carn_y, c=carn_fitness, s=120, alpha=0.8, 
                                  label='Neural Carnivores', marker='^', cmap='Reds', 
                                  vmin=0, vmax=max(30, max(carn_fitness) if carn_fitness else 30))
    
    ax1.set_xlim(0, env.width)
    ax1.set_ylim(0, env.height)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Statistics display
    stats = env.get_neural_stats()
    stats_text = f"Step: {stats['time_step']}\n"
    stats_text += f"Herbivores: {stats['herbivore_count']}\n"
    stats_text += f"Carnivores: {stats['carnivore_count']}\n"
    stats_text += f"Avg Fitness: {stats.get('avg_neural_fitness', 0):.1f}\n"
    stats_text += f"Food Available: {stats['available_food']}"
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Fitness distribution
    ax2.set_title('Neural Network Fitness Distribution', fontweight='bold')
    if neural_agents:
        all_fitness = [agent.brain.fitness_score for agent in neural_agents]
        ax2.hist(all_fitness, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(all_fitness), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_fitness):.1f}')
        ax2.set_xlabel('Fitness Score')
        ax2.set_ylabel('Number of Agents')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Energy vs Age scatter
    ax3.set_title('Energy vs Age (Neural Agents)', fontweight='bold')
    if neural_agents:
        ages = [agent.age for agent in neural_agents]
        energies = [agent.energy for agent in neural_agents]
        species_colors = ['green' if agent.species_type == SpeciesType.HERBIVORE else 'red' 
                         for agent in neural_agents]
        
        ax3.scatter(ages, energies, c=species_colors, alpha=0.7, s=60)
        ax3.set_xlabel('Age (steps)')
        ax3.set_ylabel('Current Energy')
        ax3.grid(True, alpha=0.3)
        
        # Add species legend
        ax3.scatter([], [], c='green', label='Herbivores', s=60)
        ax3.scatter([], [], c='red', label='Carnivores', s=60)
        ax3.legend()
    
    # Decision-making activity
    ax4.set_title('Neural Decision Activity', fontweight='bold')
    if neural_agents:
        decisions = [agent.brain.decisions_made for agent in neural_agents]
        fitness_scores = [agent.brain.fitness_score for agent in neural_agents]
        
        ax4.scatter(decisions, fitness_scores, alpha=0.7, s=60, c='purple')
        ax4.set_xlabel('Decisions Made')
        ax4.set_ylabel('Fitness Score')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line if enough data
        if len(decisions) > 3:
            z = np.polyfit(decisions, fitness_scores, 1)
            p = np.poly1d(z)
            ax4.plot(decisions, p(decisions), "r--", alpha=0.8, label='Trend')
            ax4.legend()
    
    plt.tight_layout()
    
    # Always save the plot in WSL environment
    if not save_path:
        save_path = f"neural_ecosystem_step_{env.time_step}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Neural visualization saved to {save_path}")
    
    # Don't show plot in headless environment
    print("ℹ️  Visualization complete - plot saved as file (WSL doesn't support GUI display)")

def plot_neural_learning_curve(env: NeuralEnvironment, steps: int = 300):
    """Run simulation and plot learning curve"""
    fitness_history = []
    population_history = []
    step_history = []
    
    print("🧠 Tracking Neural Learning Progress...")
    
    for step in range(steps):
        env.step()
        
        if step % 10 == 0:  # Sample every 10 steps
            stats = env.get_neural_stats()
            fitness_history.append(stats.get('avg_neural_fitness', 0))
            population_history.append(stats['total_population'])
            step_history.append(step)
            
            if step % 50 == 0:
                print(f"Step {step}: Fitness={stats.get('avg_neural_fitness', 0):.1f}, "
                      f"Pop={stats['total_population']}")
    
    # Create learning curve plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Fitness over time
    ax1.plot(step_history, fitness_history, 'b-', linewidth=2, label='Average Fitness')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Neural Network Fitness')
    ax1.set_title('Neural Network Learning Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add moving average
    if len(fitness_history) > 10:
        window = min(10, len(fitness_history) // 3)
        moving_avg = np.convolve(fitness_history, np.ones(window)/window, mode='valid')
        moving_steps = step_history[window-1:]
        ax1.plot(moving_steps, moving_avg, 'r--', linewidth=2, label=f'Moving Average ({window} steps)')
        ax1.legend()
    
    # Population over time
    ax2.plot(step_history, population_history, 'g-', linewidth=2, label='Total Population')
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Population Count')
    ax2.set_title('Population Dynamics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save learning curve plot
    learning_curve_path = f"neural_learning_curve_{steps}_steps.png"
    plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
    print(f"✅ Learning curve saved to {learning_curve_path}")
    print("ℹ️  Learning curve complete - plot saved as file (WSL doesn't support GUI display)")
    
    return step_history, fitness_history, population_history

if __name__ == "__main__":
    print("🎨 Neural Network Ecosystem Visualizer")
    print("=" * 50)
    
    # Create neural environment and run for a bit
    env = NeuralEnvironment()
    
    # Run simulation for 100 steps
    for _ in range(100):
        env.step()
    
    # Create visualization
    visualize_neural_ecosystem(env)
    
    # Create learning curve
    env = NeuralEnvironment()  # Fresh environment
    plot_neural_learning_curve(env, steps=200)
