"""
Simple terminal-based ecosystem monitor
"""
from src.core.ecosystem import Environment
import time
import os

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_ecosystem_map(env: Environment, width: int = 40, height: int = 20):
    """Display a simple ASCII map of the ecosystem"""
    # Create a grid to represent the ecosystem
    grid = [['.' for _ in range(width)] for _ in range(height)]
    
    # Scale positions to fit the display grid
    x_scale = env.width / width
    y_scale = env.height / height
    
    # Add food sources
    for food in env.food_sources:
        if food.is_available:
            x = int(food.position.x / x_scale)
            y = int(food.position.y / y_scale)
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = '+'
    
    # Add agents
    for agent in env.agents:
        if agent.is_alive:
            x = int(agent.position.x / x_scale)
            y = int(agent.position.y / y_scale)
            if 0 <= x < width and 0 <= y < height:
                if agent.species_type.value == "herbivore":
                    grid[y][x] = 'H'
                else:
                    grid[y][x] = 'C'
    
    # Print the grid
    print("🗺️  Ecosystem Map (H=Herbivore, C=Carnivore, +=Food, .=Empty)")
    print("┌" + "─" * width + "┐")
    for row in grid:
        print("│" + "".join(row) + "│")
    print("└" + "─" * width + "┘")

def monitor_ecosystem(steps: int = 500, update_interval: float = 0.2):
    """Monitor the ecosystem with a real-time terminal display"""
    env = Environment()
    
    print("🔬 Starting Ecosystem Monitor")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        for step in range(steps):
            clear_screen()
            
            env.step()
            stats = env.get_population_stats()
            
            print(f"🌱 AI Ecosystem Monitor - Step {stats['time_step']}")
            print("=" * 60)
            
            # Display statistics
            print(f"📊 Population:")
            print(f"   🦌 Herbivores: {stats['herbivore_count']:2d} (avg energy: {stats['avg_herbivore_energy']:5.1f})")
            print(f"   🐺 Carnivores: {stats['carnivore_count']:2d} (avg energy: {stats['avg_carnivore_energy']:5.1f})")
            print(f"   🌾 Available Food: {stats['available_food']:2d}")
            print(f"   👥 Total Population: {stats['total_population']}")
            print()
            
            # Display the map
            display_ecosystem_map(env)
            
            # Check for extinction
            if stats['total_population'] == 0:
                print("\n💀 All species extinct! Ecosystem collapsed.")
                break
            elif stats['herbivore_count'] == 0:
                print("\n🦌 Herbivores extinct! Carnivores will starve...")
            elif stats['carnivore_count'] == 0:
                print("\n🐺 Carnivores extinct! Herbivore population unbalanced...")
            
            print(f"\n⏱️  Step {step + 1}/{steps} - Press Ctrl+C to stop")
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        
    print(f"\n🏁 Final ecosystem state:")
    final_stats = env.get_population_stats()
    print(f"   Herbivores: {final_stats['herbivore_count']}")
    print(f"   Carnivores: {final_stats['carnivore_count']}")
    print(f"   Survival time: {final_stats['time_step']} steps")

if __name__ == "__main__":
    monitor_ecosystem(steps=1000, update_interval=0.3)
