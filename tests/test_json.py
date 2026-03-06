#!/usr/bin/env python3
"""
Quick JSON Serialization Test for EA-NN GPU
============================================

Test if the web interface JSON serialization issue is resolved.
"""

import os
import sys
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_json_serialization():
    """Test JSON serialization with GPU environment"""
    print("🧪 JSON Serialization Test")
    print("=" * 30)
    
    try:
        # Import and create GPU environment
        from main import Phase2NeuralEnvironment, EnhancedEcosystemWrapper, AdvancedGeneticAlgorithm, AdvancedEvolutionConfig
        
        print("✅ Creating GPU environment...")
        env = Phase2NeuralEnvironment(width=50, height=50, use_gpu=True, use_neural_agents=True)
        
        print("✅ Creating evolution wrapper...")
        genetic_config = AdvancedEvolutionConfig()
        genetic_algorithm = AdvancedGeneticAlgorithm(genetic_config)
        
        class TestEvolutionSystem:
            def __init__(self, environment, genetic_algorithm):
                self.environment = environment
                self.genetic_algorithm = genetic_algorithm
                self.generation = 0
                self.total_steps = 0
        
        evolution_system = TestEvolutionSystem(env, genetic_algorithm)
        wrapper = EnhancedEcosystemWrapper(evolution_system)
        
        print("✅ Testing agent data serialization...")
        agent_data = wrapper.get_agent_data()
        
        # Test JSON serialization
        json_data = json.dumps(agent_data)
        print(f"✅ Agent data JSON serialized successfully ({len(json_data)} chars)")
        
        print("✅ Testing stats serialization...")
        stats = wrapper.get_stats()
        json_stats = json.dumps(stats)
        print(f"✅ Stats JSON serialized successfully ({len(json_stats)} chars)")
        
        print("\n🎉 All JSON serialization tests passed!")
        print("Web interface should now work without errors.")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎮 Testing JSON serialization for RTX 2060 GPU acceleration\n")
    
    success = test_json_serialization()
    
    if success:
        print("\n🚀 Ready to run: ./run_gpu.sh --web")
    else:
        print("\n❌ JSON serialization needs attention")
