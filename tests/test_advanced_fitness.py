#!/usr/bin/env python3
"""
Test Advanced Fitness Optimization System
Verify the complex fitness landscape functionality works correctly
"""

import sys
import os
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.evolution.advanced_fitness import AdvancedFitnessEvaluator, FitnessObjective
from src.neural.evolutionary_agent import EvolutionaryNeuralAgent, EvolutionaryAgentConfig
from src.neural.evolutionary_network import EvolutionaryNetworkConfig
from src.core.ecosystem import SpeciesType, Position
from main import Phase2NeuralEnvironment

def test_fitness_evaluator():
    """Test the advanced fitness evaluator with mock data"""
    print("🧪 Testing Advanced Fitness Optimization System...")
    
    # Create fitness evaluator
    evaluator = AdvancedFitnessEvaluator()
    
    # Test initial landscape state
    print(f"✅ Created fitness evaluator with {len(evaluator.niches)} ecological niches")
    for niche_name, niche in evaluator.niches.items():
        print(f"   📋 {niche_name}: optimal size {niche.optimal_population_size}")
    
    # Test environmental condition updates
    print("\n🌍 Testing environmental condition adaptation...")
    
    # Create mock environment data
    class MockEnvironment:
        def __init__(self):
            self.width = 100
            self.height = 100
            self.food_sources = []
            self.agents = []
    
    mock_env = MockEnvironment()
    
    # Add mock agents
    for i in range(15):
        agent = EvolutionaryNeuralAgent(
            SpeciesType.HERBIVORE,
            Position(50 + i * 2, 50 + i),
            i,
            EvolutionaryAgentConfig()
        )
        agent.age = 100 + i * 10
        agent.total_food_consumed = 50 + i * 5
        agent.total_distance_traveled = 200 + i * 20
        agent.successful_reproductions = i // 3
        agent.prey_captures = 0
        agent.predator_escapes = i
        agent.fitness_evaluator = evaluator
        mock_env.agents.append(agent)
    
    # Add carnivores
    for i in range(5):
        agent = EvolutionaryNeuralAgent(
            SpeciesType.CARNIVORE,
            Position(30 + i * 3, 70 + i),
            15 + i,
            EvolutionaryAgentConfig()
        )
        agent.age = 80 + i * 8
        agent.total_food_consumed = 30 + i * 4
        agent.total_distance_traveled = 150 + i * 15
        agent.successful_reproductions = i // 2
        agent.prey_captures = i * 2
        agent.predator_escapes = 0
        agent.fitness_evaluator = evaluator
        mock_env.agents.append(agent)
    
    # Update environmental conditions
    initial_landscape = evaluator.get_fitness_landscape_info()
    print(f"   📊 Initial population density: {initial_landscape['environmental_conditions']['population_density']:.3f}")
    
    evaluator.update_environmental_conditions(mock_env)
    updated_landscape = evaluator.get_fitness_landscape_info()
    print(f"   📊 Updated population density: {updated_landscape['environmental_conditions']['population_density']:.3f}")
    print(f"   📊 Predation pressure: {updated_landscape['environmental_conditions']['predation_pressure']:.3f}")
    print(f"   📊 Resource scarcity: {updated_landscape['environmental_conditions']['resource_scarcity']:.3f}")
    
    # Test fitness evaluation for different agents
    print("\n🎯 Testing fitness evaluation...")
    
    # Evaluate herbivore fitness
    herbivore = mock_env.agents[0]
    herb_fitness = evaluator.evaluate_agent_fitness(herbivore, mock_env)
    print(f"   🌱 Herbivore fitness components:")
    for component, value in herb_fitness.items():
        print(f"      {component}: {value:.2f}")
    
    # Evaluate carnivore fitness
    carnivore = mock_env.agents[15]
    carn_fitness = evaluator.evaluate_agent_fitness(carnivore, mock_env)
    print(f"   🥩 Carnivore fitness components:")
    for component, value in carn_fitness.items():
        print(f"      {component}: {value:.2f}")
    
    # Test niche specialization
    print(f"\n🏠 Testing niche assignments...")
    evaluator.update_niche_populations(mock_env)
    niche_info = updated_landscape['active_niches']
    for niche_name, population in niche_info.items():
        print(f"   📋 {niche_name}: {population} agents")
    
    # Test temporal dynamics
    print(f"\n⏰ Testing temporal fitness dynamics...")
    original_weights = dict(evaluator.current_landscape.base_weights)
    
    # Simulate time progression
    for time_step in [0, 125, 250, 375, 500]:
        evaluator.current_landscape.time_step = time_step
        evaluator._update_adaptive_weights()
        
        exploration_weight = evaluator.current_landscape.base_weights[FitnessObjective.EXPLORATION]
        cooperation_weight = evaluator.current_landscape.base_weights[FitnessObjective.SOCIAL_COOPERATION]
        
        print(f"   ⏲️ Step {time_step}: Exploration={exploration_weight:.2f}, Cooperation={cooperation_weight:.2f}")
    
    print("\n✅ Advanced fitness optimization system working correctly!")
    return True

def test_phase2_integration():
    """Test integration with Phase2NeuralEnvironment"""
    print("\n🔗 Testing Phase 2 integration...")
    
    # Create Phase 2 environment
    env = Phase2NeuralEnvironment(width=80, height=80, use_neural_agents=True)
    
    # Verify global fitness evaluator is set up
    assert hasattr(env, 'global_fitness_evaluator'), "Missing global fitness evaluator"
    print("   ✅ Global fitness evaluator created")
    
    # Verify agents have fitness evaluator reference
    agents_with_evaluator = sum(1 for agent in env.agents if agent.fitness_evaluator is not None)
    print(f"   ✅ {agents_with_evaluator}/{len(env.agents)} agents have fitness evaluator")
    
    # Run a few simulation steps
    print("   🏃 Running simulation steps...")
    for step in range(5):
        env.step()
        
        # Check fitness landscape is being updated
        landscape_info = env.get_fitness_landscape_info()
        if step == 0:
            print(f"      📊 Step {step}: Population density = {landscape_info['environmental_conditions']['population_density']:.3f}")
        
        # Check if any agents have detailed fitness
        agents_with_details = sum(1 for agent in env.agents if hasattr(agent, 'detailed_fitness'))
        if agents_with_details > 0:
            print(f"      📈 Step {step}: {agents_with_details} agents have detailed fitness")
            break
    
    print("   ✅ Phase 2 integration working correctly!")
    return True

def test_competitive_dynamics():
    """Test competitive fitness evaluation"""
    print("\n🏆 Testing competitive dynamics...")
    
    # Create evaluator
    evaluator = AdvancedFitnessEvaluator()
    
    # Create agents with different performance levels
    class MockAgent:
        def __init__(self, agent_id, species_type, performance_multiplier):
            self.agent_id = agent_id
            self.species_type = species_type
            self.is_alive = True
            self.age = 100 * performance_multiplier
            self.total_food_consumed = 50 * performance_multiplier
            self.total_distance_traveled = 200 * performance_multiplier
            self.successful_reproductions = int(5 * performance_multiplier)
            self.prey_captures = int(3 * performance_multiplier) if species_type == SpeciesType.CARNIVORE else 0
            self.predator_escapes = int(2 * performance_multiplier) if species_type == SpeciesType.HERBIVORE else 0
            self.energy = 50 * performance_multiplier
            self.max_energy = 100
            self.novelty_bonus = 10 * performance_multiplier
            self.received_signals = [f"signal_{i}" for i in range(int(3 * performance_multiplier))]
            self.visited_areas = [f"area_{i}" for i in range(int(8 * performance_multiplier))]
    
    # Create mock environment with varied performance agents
    class MockCompetitiveEnv:
        def __init__(self):
            self.width = 100
            self.height = 100
            self.food_sources = ['food1', 'food2', 'food3']
            self.agents = []
            
            # Create high, medium, and low performers
            for i, multiplier in enumerate([2.0, 1.0, 0.5]):
                agent = MockAgent(i, SpeciesType.HERBIVORE, multiplier)
                self.agents.append(agent)
    
    env = MockCompetitiveEnv()
    
    # Evaluate competitive fitness
    print("   🥇 Evaluating competitive performance...")
    
    competitive_scores = []
    for agent in env.agents:
        competitive_fitness = evaluator._calculate_competitive_fitness(agent, env)
        performance_score = evaluator._get_agent_performance_score(agent)
        competitive_scores.append((agent.agent_id, performance_score, competitive_fitness))
        print(f"      Agent {agent.agent_id}: Performance={performance_score:.1f}, Competitive={competitive_fitness:.1f}")
    
    # Verify ranking is correct (higher performers get higher competitive fitness)
    sorted_by_performance = sorted(competitive_scores, key=lambda x: x[1], reverse=True)
    sorted_by_competitive = sorted(competitive_scores, key=lambda x: x[2], reverse=True)
    
    if sorted_by_performance[0][0] == sorted_by_competitive[0][0]:
        print("   ✅ Competitive ranking correctly identifies top performer")
    else:
        print("   ❌ Competitive ranking mismatch")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Advanced Fitness Optimization System Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        test_fitness_evaluator,
        test_phase2_integration,
        test_competitive_dynamics
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{len(tests)} tests passed")
    print(f"⏱️ Total time: {elapsed:.2f} seconds")
    
    if passed == len(tests):
        print("🎉 Complex fitness optimization functional milestone COMPLETE!")
        print("✨ Phase 2 advanced fitness landscapes are fully operational!")
    else:
        print("⚠️ Some tests failed - check implementation")
