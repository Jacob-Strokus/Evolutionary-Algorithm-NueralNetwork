#!/usr/bin/env python3
"""
Test Generation Tracking System
Verify that generation information is properly tracked and displayed
"""

import sys
import os
# Add parent directory to path to access src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType

def test_generation_tracking():
    """Test that agents properly track their generation"""
    print("🧬 Testing Generation Tracking System")
    print("=" * 50)
    
    # Create environment
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Get initial agents
    neural_agents = [agent for agent in env.agents if isinstance(agent, NeuralAgent)]
    herbivores = [a for a in neural_agents if a.species_type == SpeciesType.HERBIVORE]
    carnivores = [a for a in neural_agents if a.species_type == SpeciesType.CARNIVORE]
    
    print(f"🦌 Initial Herbivores: {len(herbivores)}")
    print(f"🐺 Initial Carnivores: {len(carnivores)}")
    print()
    
    # Check initial generation values
    print("📊 Initial Generation Values:")
    for i, herb in enumerate(herbivores[:3]):  # Check first 3
        gen = getattr(herb, 'generation', 'MISSING')
        print(f"   Herbivore {i}: Generation {gen}")
    
    for i, carn in enumerate(carnivores[:3]):  # Check first 3
        gen = getattr(carn, 'generation', 'MISSING')
        print(f"   Carnivore {i}: Generation {gen}")
    
    print()
    
    # Force reproduction to test generation inheritance
    print("🔄 Testing Reproduction and Generation Inheritance:")
    
    if herbivores:
        parent = herbivores[0]
        parent.energy = 200  # Ensure it can reproduce
        parent.reproduction_cooldown = 0
        
        print(f"   Parent herbivore generation: {getattr(parent, 'generation', 'MISSING')}")
        
        # Run a few steps to trigger reproduction
        initial_count = len([a for a in env.agents if isinstance(a, NeuralAgent)])
        
        for step in range(50):
            env.step()
            current_count = len([a for a in env.agents if isinstance(a, NeuralAgent)])
            
            if current_count > initial_count:
                # Reproduction occurred!
                new_agents = [a for a in env.agents if isinstance(a, NeuralAgent)]
                print(f"   🎉 Reproduction occurred at step {step}!")
                print(f"   Population increased from {initial_count} to {current_count}")
                
                # Find newest agents
                herbivores_new = [a for a in new_agents if a.species_type == SpeciesType.HERBIVORE]
                
                # Check generations of newest agents
                for herb in herbivores_new[-3:]:  # Check last 3 herbivores
                    gen = getattr(herb, 'generation', 'MISSING')
                    print(f"   New herbivore generation: {gen}")
                
                break
        else:
            print("   ⚠️ No reproduction occurred in 50 steps")
    
    print()
    
    # Test web server data structure
    print("🌐 Testing Web Server Data Structure:")
    try:
        from src.visualization.realtime_web_server import EcosystemWebServer
        from src.visualization.visualizer import EcosystemCanvas
        
        canvas = EcosystemCanvas(width=400, height=400)
        canvas.create_neural_environment()
        server = EcosystemWebServer(canvas)
        
        # Get current data
        data = server.get_current_data()
        
        if 'agents' in data:
            if 'herbivores' in data['agents'] and 'generations' in data['agents']['herbivores']:
                herb_gens = data['agents']['herbivores']['generations']
                print(f"   ✅ Herbivore generations in data: {herb_gens[:5]}")
            else:
                print("   ❌ Herbivore generations missing from data")
            
            if 'carnivores' in data['agents'] and 'generations' in data['agents']['carnivores']:
                carn_gens = data['agents']['carnivores']['generations']
                print(f"   ✅ Carnivore generations in data: {carn_gens[:5]}")
            else:
                print("   ❌ Carnivore generations missing from data")
        
        # Test agent details
        agents = [a for a in canvas.env.agents if isinstance(a, NeuralAgent)]
        if agents:
            herbivore = next((a for a in agents if a.species_type == SpeciesType.HERBIVORE), None)
            if herbivore:
                details = server.get_agent_details('herb_0')
                if 'generation' in details:
                    print(f"   ✅ Agent details include generation: {details['generation']}")
                else:
                    print("   ❌ Agent details missing generation")
        
    except Exception as e:
        print(f"   ❌ Web server test failed: {e}")
    
    print()
    print("🏁 Generation Tracking Test Complete!")

if __name__ == "__main__":
    test_generation_tracking()
