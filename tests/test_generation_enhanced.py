#!/usr/bin/env python3
"""
Enhanced Generation Tracking Test
Test reproduction with proper ecosystem mechanics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType

def test_actual_reproduction():
    """Test reproduction in a more realistic scenario"""
    print("🧬 Testing Actual Reproduction Events")
    print("=" * 50)
    
    # Create larger environment with longer simulation
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Track initial state
    initial_agents = [a for a in env.agents if isinstance(a, NeuralAgent)]
    initial_count = len(initial_agents)
    
    print(f"📊 Initial population: {initial_count}")
    print(f"   Herbivores: {len([a for a in initial_agents if a.species_type == SpeciesType.HERBIVORE])}")
    print(f"   Carnivores: {len([a for a in initial_agents if a.species_type == SpeciesType.CARNIVORE])}")
    
    # Track generation statistics over time
    generation_stats = {}
    reproduction_events = []
    
    print("\n🔄 Running extended simulation to observe natural reproduction...")
    print("Step | Population | New Agents | Generations Present")
    print("-" * 55)
    
    for step in range(500):  # Longer simulation
        env.step()
        
        current_agents = [a for a in env.agents if isinstance(a, NeuralAgent)]
        current_count = len(current_agents)
        
        # Check for new agents (reproduction occurred)
        if current_count > initial_count:
            new_agents = current_agents[initial_count:]
            reproduction_events.append({
                'step': step,
                'new_count': len(new_agents),
                'total_population': current_count
            })
            
            # Analyze generations of new agents
            for agent in new_agents:
                gen = getattr(agent, 'generation', 1)
                if gen not in generation_stats:
                    generation_stats[gen] = 0
                generation_stats[gen] += 1
            
            initial_count = current_count  # Update for next iteration
        
        # Report every 100 steps
        if step % 100 == 0:
            generations_present = set()
            for agent in current_agents:
                gen = getattr(agent, 'generation', 1)
                generations_present.add(gen)
            
            new_agents_this_period = len([e for e in reproduction_events if e['step'] > step - 100])
            
            print(f"{step:4d} | {current_count:10d} | {new_agents_this_period:10d} | {sorted(generations_present)}")
    
    print(f"\n📈 Reproduction Analysis:")
    print(f"   Total reproduction events: {len(reproduction_events)}")
    print(f"   Generation distribution: {generation_stats}")
    
    if reproduction_events:
        print("   ✅ Natural reproduction occurred!")
        
        # Check if higher generations were created
        max_generation = max(generation_stats.keys()) if generation_stats else 1
        print(f"   Highest generation reached: {max_generation}")
        
        if max_generation > 1:
            print("   ✅ Generation inheritance working in natural reproduction!")
            return True
        else:
            print("   ❓ Only generation 1 observed - may need longer simulation")
            return False
    else:
        print("   ⚠️ No natural reproduction in 500 steps")
        return False

def test_web_interface_data():
    """Test that generation data is properly formatted for web interface"""
    print("\n🌐 Testing Web Interface Data Structure")
    print("=" * 45)
    
    try:
        env = NeuralEnvironment(width=60, height=60, use_neural_agents=True)
        
        # Simulate some reproduction by manually creating agents with different generations
        from src.core.ecosystem import Position
        import random
        
        # Create a generation 2 agent manually
        test_pos = Position(30, 30)
        gen2_agent = NeuralAgent(SpeciesType.HERBIVORE, test_pos, env.next_agent_id, generation=2)
        env.agents.append(gen2_agent)
        env.next_agent_id += 1
        
        # Test web server data collection
        from src.visualization.realtime_web_server import EcosystemWebServer
        
        # Mock canvas object for testing
        class MockCanvas:
            def __init__(self, env):
                self.env = env
                self.step_count = 0
                self.fitness_history_herb = []
                self.fitness_history_carn = []
                self.population_history = []
            
            def get_agent_data(self):
                neural_agents = [a for a in self.env.agents if isinstance(a, NeuralAgent)]
                herbivores = [a for a in neural_agents if a.species_type == SpeciesType.HERBIVORE]
                carnivores = [a for a in neural_agents if a.species_type == SpeciesType.CARNIVORE]
                
                return {
                    'herbivores': {
                        'x': [h.position.x for h in herbivores],
                        'y': [h.position.y for h in herbivores],
                        'fitness': [h.brain.fitness_score for h in herbivores]
                    },
                    'carnivores': {
                        'x': [c.position.x for c in carnivores],
                        'y': [c.position.y for c in carnivores],
                        'fitness': [c.brain.fitness_score for c in carnivores]
                    },
                    'food': {'x': [], 'y': []}
                }
        
        mock_canvas = MockCanvas(env)
        server = EcosystemWebServer(mock_canvas)
        
        # Test data structure
        data = server.get_current_data()
        
        print("📊 Web Interface Data Check:")
        
        # Check herbivore generations
        if 'agents' in data and 'herbivores' in data['agents']:
            if 'generations' in data['agents']['herbivores']:
                herb_gens = data['agents']['herbivores']['generations']
                print(f"   ✅ Herbivore generations: {herb_gens}")
                
                if 2 in herb_gens:
                    print("   ✅ Generation 2 agent detected in web data!")
                else:
                    print("   ❓ Generation 2 agent not found in web data")
            else:
                print("   ❌ Generation data missing from herbivores")
                return False
        
        # Test agent details
        neural_agents = [a for a in env.agents if isinstance(a, NeuralAgent)]
        herbivores = [a for a in neural_agents if a.species_type == SpeciesType.HERBIVORE]
        
        if herbivores:
            # Find our generation 2 agent
            gen2_agents = [h for h in herbivores if getattr(h, 'generation', 1) == 2]
            if gen2_agents:
                details = server.get_agent_details('herb_' + str(herbivores.index(gen2_agents[0])))
                if 'generation' in details and details['generation'] == 2:
                    print("   ✅ Agent inspection includes generation info!")
                    return True
                else:
                    print("   ❌ Agent details missing generation")
                    return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Web interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def comprehensive_test():
    """Run all generation tests comprehensively"""
    print("\n🧬 Comprehensive Generation System Test")
    print("=" * 50)
    
    results = {
        'basic_assignment': False,
        'reproduction_mechanics': False,
        'web_interface': False
    }
    
    # Test 1: Basic assignment
    env = NeuralEnvironment(width=40, height=40, use_neural_agents=True)
    neural_agents = [a for a in env.agents if isinstance(a, NeuralAgent)]
    
    all_gen_1 = all(getattr(a, 'generation', 0) == 1 for a in neural_agents)
    results['basic_assignment'] = all_gen_1
    
    print(f"✅ Basic Assignment: {'PASS' if all_gen_1 else 'FAIL'}")
    
    # Test 2: Reproduction mechanics (already tested)
    results['reproduction_mechanics'] = test_actual_reproduction()
    
    # Test 3: Web interface
    results['web_interface'] = test_web_interface_data()
    
    print(f"\n🏁 Comprehensive Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    if all(results.values()):
        print("\n🎉 All generation tracking systems working perfectly!")
    elif results['basic_assignment'] and results['web_interface']:
        print("\n✅ Core generation tracking ready for production!")
        print("⚠️ Extended reproduction testing may need longer simulation time")
    else:
        print("\n⚠️ Some generation tracking features need attention")

if __name__ == "__main__":
    print("🧬 Enhanced Generation Tracking Test Suite")
    print("=" * 70)
    
    try:
        comprehensive_test()
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
