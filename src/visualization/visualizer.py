"""
Visualization module for the AI Ecosystem Simulation
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.core.ecosystem import Environment, SpeciesType
import numpy as np

# Set matplotlib to use non-interactive backend for WSL/headless environments
import matplotlib
matplotlib.use('Agg')

def plot_population_over_time(env: Environment, steps: int = 500):
    """Run simulation and plot population changes over time"""
    herbivore_counts = []
    carnivore_counts = []
    food_counts = []
    time_steps = []
    
    print("🔬 Running simulation and collecting data...")
    
    for step in range(steps):
        env.step()
        stats = env.get_population_stats()
        
        time_steps.append(stats['time_step'])
        herbivore_counts.append(stats['herbivore_count'])
        carnivore_counts.append(stats['carnivore_count'])
        food_counts.append(stats['available_food'])
        
        if step % 50 == 0:
            print(f"Step {step}: H={stats['herbivore_count']}, C={stats['carnivore_count']}")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Population plot
    ax1.plot(time_steps, herbivore_counts, 'green', label='Herbivores', linewidth=2)
    ax1.plot(time_steps, carnivore_counts, 'red', label='Carnivores', linewidth=2)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Population')
    ax1.set_title('Population Dynamics Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Food availability plot
    ax2.plot(time_steps, food_counts, 'brown', label='Available Food', linewidth=2)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Available Food Sources')
    ax2.set_title('Food Availability Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return time_steps, herbivore_counts, carnivore_counts, food_counts

def visualize_ecosystem_snapshot(env: Environment, save_path: str = None):
    """Create a static visualization of the current ecosystem state"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot food sources
    food_x = [food.position.x for food in env.food_sources if food.is_available]
    food_y = [food.position.y for food in env.food_sources if food.is_available]
    ax.scatter(food_x, food_y, c='brown', s=30, alpha=0.6, label='Food', marker='s')
    
    # Plot herbivores
    herbivore_x = [agent.position.x for agent in env.agents 
                   if agent.species_type == SpeciesType.HERBIVORE and agent.is_alive]
    herbivore_y = [agent.position.y for agent in env.agents 
                   if agent.species_type == SpeciesType.HERBIVORE and agent.is_alive]
    herbivore_energies = [agent.energy for agent in env.agents 
                         if agent.species_type == SpeciesType.HERBIVORE and agent.is_alive]
    
    if herbivore_x:
        scatter_h = ax.scatter(herbivore_x, herbivore_y, c=herbivore_energies, 
                              s=60, alpha=0.8, label='Herbivores', marker='o', 
                              cmap='Greens', vmin=0, vmax=120)
    
    # Plot carnivores
    carnivore_x = [agent.position.x for agent in env.agents 
                   if agent.species_type == SpeciesType.CARNIVORE and agent.is_alive]
    carnivore_y = [agent.position.y for agent in env.agents 
                   if agent.species_type == SpeciesType.CARNIVORE and agent.is_alive]
    carnivore_energies = [agent.energy for agent in env.agents 
                         if agent.species_type == SpeciesType.CARNIVORE and agent.is_alive]
    
    if carnivore_x:
        scatter_c = ax.scatter(carnivore_x, carnivore_y, c=carnivore_energies, 
                              s=100, alpha=0.8, label='Carnivores', marker='^', 
                              cmap='Reds', vmin=0, vmax=150)
    
    # Add colorbars
    if herbivore_x:
        plt.colorbar(scatter_h, ax=ax, label='Herbivore Energy', shrink=0.5, pad=0.1)
    if carnivore_x:
        plt.colorbar(scatter_c, ax=ax, label='Carnivore Energy', shrink=0.5, pad=0.15)
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Ecosystem Snapshot - Step {env.time_step}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats = env.get_population_stats()
    stats_text = f"Herbivores: {stats['herbivore_count']}\n"
    stats_text += f"Carnivores: {stats['carnivore_count']}\n"
    stats_text += f"Available Food: {stats['available_food']}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def create_animated_ecosystem(env: Environment, steps: int = 200, interval: int = 100):
    """Create an animated visualization of the ecosystem"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def animate(frame):
        ax.clear()
        
        # Run one simulation step
        env.step()
        
        # Plot food sources
        food_x = [food.position.x for food in env.food_sources if food.is_available]
        food_y = [food.position.y for food in env.food_sources if food.is_available]
        ax.scatter(food_x, food_y, c='brown', s=20, alpha=0.6, marker='s')
        
        # Plot herbivores
        herbivore_x = [agent.position.x for agent in env.agents 
                       if agent.species_type == SpeciesType.HERBIVORE and agent.is_alive]
        herbivore_y = [agent.position.y for agent in env.agents 
                       if agent.species_type == SpeciesType.HERBIVORE and agent.is_alive]
        
        if herbivore_x:
            ax.scatter(herbivore_x, herbivore_y, c='green', s=40, alpha=0.8, marker='o')
        
        # Plot carnivores
        carnivore_x = [agent.position.x for agent in env.agents 
                       if agent.species_type == SpeciesType.CARNIVORE and agent.is_alive]
        carnivore_y = [agent.position.y for agent in env.agents 
                       if agent.species_type == SpeciesType.CARNIVORE and agent.is_alive]
        
        if carnivore_x:
            ax.scatter(carnivore_x, carnivore_y, c='red', s=60, alpha=0.8, marker='^')
        
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_title(f'Ecosystem Animation - Step {env.time_step}')
        
        # Add statistics
        stats = env.get_population_stats()
        stats_text = f"H: {stats['herbivore_count']} | C: {stats['carnivore_count']} | F: {stats['available_food']}"
        ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, 
                horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ani = animation.FuncAnimation(fig, animate, frames=steps, interval=interval, repeat=False)
    plt.show()
    
    return ani

def demonstrate_visualizations():
    """Run demonstration of all visualization capabilities"""
    print("🎨 Ecosystem Visualization Demonstration")
    print("=" * 50)
    
    # Create environment and run simulation
    print("📊 Creating ecosystem and running simulation...")
    env = Environment()
    
    # Run simulation for enough steps to get interesting data
    print("🏃 Running 300 simulation steps...")
    for step in range(300):
        env.step()
        if step % 50 == 0:
            stats = env.get_population_stats()
            print(f"  Step {step}: H={stats['herbivore_count']}, C={stats['carnivore_count']}")
    
    print("\n📈 Generating visualizations...")
    
    # 1. Population over time
    print("1. Creating population dynamics chart...")
    env_plot = Environment()
    plot_population_over_time(env_plot, steps=200)
    print("   ✅ Population chart saved!")
    
    # 2. Ecosystem snapshot
    print("2. Creating ecosystem snapshot...")
    visualize_ecosystem_snapshot(env, "ecosystem_snapshot_demo.png")
    print("   ✅ Ecosystem snapshot saved!")
    
    print("\n🎉 Visualization demonstration complete!")
    print("📁 Check the current directory for generated plots:")
    print("   - population_dynamics.png")
    print("   - ecosystem_snapshot_demo.png")

if __name__ == "__main__":
    # Example usage
    print("🎨 Testing ecosystem visualization...")
    
    # Create environment and run a few steps
    env = Environment()
    for _ in range(50):
        env.step()
    
    # Create snapshot
    visualize_ecosystem_snapshot(env)
    
    # Plot population over time
    env = Environment()
    plot_population_over_time(env, steps=300)
