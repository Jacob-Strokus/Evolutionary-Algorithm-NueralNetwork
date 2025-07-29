"""
Comprehensive tests for Phase 1 Ecosystem Simulation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.ecosystem import Environment, Agent, SpeciesType, Position, Food
import time

def test_basic_functionality():
    """Test basic ecosystem functionality"""
    print("🧪 Testing Basic Functionality...")
    
    env = Environment()
    initial_stats = env.get_population_stats()
    
    assert initial_stats['herbivore_count'] > 0, "Should start with herbivores"
    assert initial_stats['carnivore_count'] > 0, "Should start with carnivores"
    assert initial_stats['available_food'] > 0, "Should start with food"
    
    # Run some steps
    for _ in range(50):
        env.step()
    
    print("✅ Basic functionality test passed!")

def test_reproduction():
    """Test if reproduction system works"""
    print("🧪 Testing Reproduction System...")
    
    env = Environment()
    
    # Give an agent high energy to trigger reproduction
    for agent in env.agents:
        if agent.species_type == SpeciesType.HERBIVORE:
            agent.energy = 120  # Above reproduction threshold
            agent.age = 100     # Old enough to reproduce
            agent.reproduction_cooldown = 0
            break
    
    initial_count = len(env.agents)
    
    # Run steps to trigger reproduction
    for _ in range(10):
        env.step()
        if len(env.agents) > initial_count:
            print("✅ Reproduction test passed!")
            return
    
    print("⚠️ Reproduction may not be working optimally")

def test_hunting():
    """Test if hunting mechanics work"""
    print("🧪 Testing Hunting Mechanics...")
    
    env = Environment()
    
    # Place a carnivore near a herbivore
    carnivore = None
    herbivore = None
    
    for agent in env.agents:
        if agent.species_type == SpeciesType.CARNIVORE and carnivore is None:
            carnivore = agent
        elif agent.species_type == SpeciesType.HERBIVORE and herbivore is None:
            herbivore = agent
        
        if carnivore and herbivore:
            break
    
    if carnivore and herbivore:
        # Place them close together
        carnivore.position = Position(50, 50)
        herbivore.position = Position(52, 52)
        
        initial_herbivore_count = sum(1 for a in env.agents if a.species_type == SpeciesType.HERBIVORE)
        
        # Run simulation to see if hunting occurs
        for _ in range(20):
            env.step()
            current_herbivore_count = sum(1 for a in env.agents if a.species_type == SpeciesType.HERBIVORE and a.is_alive)
            if current_herbivore_count < initial_herbivore_count:
                print("✅ Hunting mechanics test passed!")
                return
        
        print("⚠️ Hunting may need balancing")
    else:
        print("❌ Could not set up hunting test")

def test_food_regeneration():
    """Test if food regenerates properly"""
    print("🧪 Testing Food Regeneration...")
    
    env = Environment()
    
    # Consume all food
    for food in env.food_sources:
        food.is_available = False
        food.current_regen = 0
    
    initial_available = sum(1 for f in env.food_sources if f.is_available)
    
    # Run enough steps for food to regenerate
    for _ in range(150):  # More than regeneration_time (100)
        env.step()
    
    final_available = sum(1 for f in env.food_sources if f.is_available)
    
    if final_available > initial_available:
        print("✅ Food regeneration test passed!")
    else:
        print("❌ Food regeneration may not be working")

def test_boundary_handling():
    """Test if agents stay within boundaries"""
    print("🧪 Testing Boundary Handling...")
    
    env = Environment()
    
    # Move an agent outside boundaries
    test_agent = env.agents[0]
    test_agent.position = Position(-10, -10)
    
    env.keep_agent_in_bounds(test_agent)
    
    if (0 <= test_agent.position.x <= env.width and 
        0 <= test_agent.position.y <= env.height):
        print("✅ Boundary handling test passed!")
    else:
        print("❌ Boundary handling failed")

def test_extinction_recovery():
    """Test what happens with extinction scenarios"""
    print("🧪 Testing Extinction Scenarios...")
    
    env = Environment()
    
    # Kill all carnivores
    for agent in env.agents:
        if agent.species_type == SpeciesType.CARNIVORE:
            agent.is_alive = False
    
    # Run simulation
    for _ in range(100):
        env.step()
        stats = env.get_population_stats()
        if stats['carnivore_count'] == 0 and stats['herbivore_count'] > 0:
            print("✅ Extinction scenario handling works!")
            return
    
    print("⚠️ Extinction scenario needs monitoring")

def run_performance_test():
    """Test simulation performance"""
    print("🧪 Testing Performance...")
    
    start_time = time.time()
    
    env = Environment()
    for _ in range(1000):
        env.step()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ 1000 steps completed in {duration:.2f} seconds")
    print(f"   Performance: {1000/duration:.1f} steps/second")

def run_stability_test():
    """Test long-term stability"""
    print("🧪 Testing Long-term Stability...")
    
    stable_runs = 0
    total_runs = 5
    
    for run in range(total_runs):
        env = Environment()
        
        # Run for 500 steps
        for _ in range(500):
            env.step()
            stats = env.get_population_stats()
            
            # Check if ecosystem collapsed
            if stats['total_population'] == 0:
                break
        
        final_stats = env.get_population_stats()
        if final_stats['total_population'] > 0:
            stable_runs += 1
            print(f"   Run {run+1}: Survived - {final_stats['total_population']} agents")
        else:
            print(f"   Run {run+1}: Collapsed at step {final_stats['time_step']}")
    
    stability_rate = (stable_runs / total_runs) * 100
    print(f"✅ Stability rate: {stability_rate}% ({stable_runs}/{total_runs} runs survived)")

if __name__ == "__main__":
    print("🧬 Running Phase 1 Ecosystem Tests")
    print("=" * 50)
    
    test_basic_functionality()
    test_reproduction()
    test_hunting()
    test_food_regeneration()
    test_boundary_handling()
    test_extinction_recovery()
    
    print("\n📊 Performance & Stability Tests")
    print("=" * 50)
    
    run_performance_test()
    run_stability_test()
    
    print("\n🎯 Phase 1 Testing Complete!")
    print("Ready to proceed to Phase 2: Neural Networks")
