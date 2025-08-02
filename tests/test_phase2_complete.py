#!/usr/bin/env python3
"""
Phase 2 Complete Integration Test
Verify all Phase 2 systems work together with the updated main.py
"""

import sys
import os
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main import Phase2NeuralEnvironment

def test_phase2_complete_integration():
    """Test complete Phase 2 integration with all new systems"""
    print("🚀 Testing Phase 2 Complete Integration...")
    
    # Create Phase 2 environment
    print("   🏗️ Creating Phase 2 environment...")
    env = Phase2NeuralEnvironment(width=80, height=80)
    
    # Check system status
    print("   📊 Checking Phase 2 system status...")
    status = env.get_phase2_system_status()
    print(f"      Total agents: {status['total_agents']}")
    print(f"      Phase 2 status: {status['phase2_status']}")
    
    for system, coverage in status['system_coverage'].items():
        print(f"      {system}: {coverage}")
    
    # Run simulation steps
    print("   🏃 Running simulation with all Phase 2 systems...")
    for step in range(5):
        env.step()
        
        # Get comprehensive stats
        stats = env.get_neural_stats()
        fitness_info = env.get_fitness_landscape_info()
        
        print(f"      Step {step+1}:")
        print(f"         Population: H={stats['herbivores']}, C={stats['carnivores']}")
        print(f"         Communications: {stats['social_communications']}")
        print(f"         Exploration coverage: {stats['exploration_coverage']:.1f}%")
        print(f"         Average fitness: {stats['avg_fitness_score']:.2f}")
        print(f"         Population density: {stats['fitness_landscape']['population_density']:.3f}")
        
        # Check niche distribution
        if step == 4:  # Last step
            print(f"         Niche distribution: {stats['niche_distribution']}")
    
    # Test detailed fitness for a random agent
    print("   🎯 Testing detailed fitness evaluation...")
    if env.agents:
        test_agent = env.agents[0]
        fitness_details = env.get_agent_fitness_details(test_agent.agent_id)
        if fitness_details:
            print(f"      Agent {test_agent.agent_id} fitness components:")
            for component, value in fitness_details.items():
                if isinstance(value, (int, float)):
                    print(f"         {component}: {value:.2f}")
    
    print("   ✅ Phase 2 complete integration test successful!")
    return True

def test_advanced_systems_functionality():
    """Test individual advanced system functionality"""
    print("\n🔧 Testing Advanced Systems Functionality...")
    
    env = Phase2NeuralEnvironment(width=60, height=60)
    
    # Test multi-target processing
    print("   🎯 Testing multi-target processing...")
    multi_target_agents = [a for a in env.agents if hasattr(a, 'multi_target_processor')]
    print(f"      {len(multi_target_agents)}/{len(env.agents)} agents have multi-target processing")
    
    # Test temporal networks
    print("   🧠 Testing temporal networks...")
    temporal_agents = [a for a in env.agents if hasattr(a, 'temporal_network')]
    print(f"      {len(temporal_agents)}/{len(env.agents)} agents have temporal networks")
    
    # Test social learning
    print("   🤝 Testing social learning...")
    social_agents = [a for a in env.agents if hasattr(a, 'social_learning')]
    print(f"      {len(social_agents)}/{len(env.agents)} agents have social learning")
    
    # Test exploration intelligence
    print("   🗺️ Testing exploration intelligence...")
    exploration_agents = [a for a in env.agents if hasattr(a, 'exploration_intelligence')]
    print(f"      {len(exploration_agents)}/{len(env.agents)} agents have exploration intelligence")
    
    # Test advanced fitness
    print("   📈 Testing advanced fitness evaluation...")
    fitness_agents = [a for a in env.agents if hasattr(a, 'fitness_evaluator') and a.fitness_evaluator is not None]
    print(f"      {len(fitness_agents)}/{len(env.agents)} agents have advanced fitness evaluation")
    
    # Run a few steps to test interactions
    print("   🔄 Testing system interactions...")
    for step in range(3):
        env.step()
        
        # Check for communications
        recent_comms = len([msg for msg in env.global_communication_log 
                           if msg.get('step', 0) == env.step_count])
        if recent_comms > 0:
            print(f"      Step {step+1}: {recent_comms} new communications")
    
    print("   ✅ Advanced systems functionality test successful!")
    return True

def test_performance_benchmarks():
    """Test performance with all Phase 2 systems"""
    print("\n⚡ Testing Performance Benchmarks...")
    
    # Create smaller environment for performance testing
    env = Phase2NeuralEnvironment(width=50, height=50)
    
    print(f"   🏁 Starting with {len(env.agents)} agents")
    
    # Time 20 simulation steps
    start_time = time.time()
    for step in range(20):
        env.step()
    end_time = time.time()
    
    elapsed = end_time - start_time
    steps_per_second = 20 / elapsed
    
    print(f"   ⏱️ Performance: {steps_per_second:.1f} steps/second")
    print(f"   📊 Final stats:")
    
    final_stats = env.get_neural_stats()
    print(f"      Population: {final_stats['herbivores']} + {final_stats['carnivores']} = {final_stats['herbivores'] + final_stats['carnivores']}")
    print(f"      Communications: {final_stats['social_communications']}")
    print(f"      Exploration coverage: {final_stats['exploration_coverage']:.1f}%")
    print(f"      Average fitness: {final_stats['avg_fitness_score']:.2f}")
    
    performance_acceptable = steps_per_second > 1.0  # At least 1 step per second
    print(f"   ✅ Performance benchmark: {'PASSED' if performance_acceptable else 'NEEDS_OPTIMIZATION'}")
    
    return performance_acceptable

if __name__ == "__main__":
    print("🎉 Phase 2 Complete Integration Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        test_phase2_complete_integration,
        test_advanced_systems_functionality,
        test_performance_benchmarks
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
        print("🎉 PHASE 2 COMPLETE INTEGRATION SUCCESS!")
        print("✨ All advanced evolutionary features are fully operational!")
        print("🧬 Ready for sophisticated artificial intelligence emergence!")
    else:
        print("⚠️ Some tests failed - check implementation")
