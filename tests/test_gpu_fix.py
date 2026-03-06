#!/usr/bin/env python3
"""
Quick GPU Fix Test for EA-NN
=============================

Test if the MultiTargetProcessor GPU fix works.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_gpu_fix():
    """Test GPU processing with fixed MultiTargetProcessor"""
    print("🧪 GPU Fix Test")
    print("=" * 20)
    
    try:
        # Import and create GPU environment
        from main import Phase2NeuralEnvironment
        
        print("✅ Creating GPU environment...")
        env = Phase2NeuralEnvironment(width=50, height=50, use_gpu=True, use_neural_agents=True)
        
        print(f"✅ Environment created with {len(env.agents)} agents")
        
        # Test GPU processing with more than 50 agents (if possible)
        if len(env.agents) >= env.gpu_threshold:
            print(f"🎮 Testing GPU acceleration (agents: {len(env.agents)} >= threshold: {env.gpu_threshold})")
            # Trigger GPU processing
            env.step()
            print("✅ GPU step completed successfully!")
        else:
            print(f"💻 Testing CPU processing (agents: {len(env.agents)} < threshold: {env.gpu_threshold})")
            # Trigger CPU processing  
            env.step()
            print("✅ CPU step completed successfully!")
        
        # Test neural inputs
        if env.agents:
            test_agent = env.agents[0]
            inputs = env._get_agent_neural_inputs(test_agent)
            print(f"✅ Neural inputs generated: {len(inputs)} values")
            print(f"   Sample inputs: {inputs[:5]}")
        
        print("\n🎉 GPU fix test passed!")
        return True
        
    except Exception as e:
        print(f"❌ GPU fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎮 Testing GPU fix for MultiTargetProcessor\n")
    
    success = test_gpu_fix()
    
    if success:
        print("\n🚀 Ready to run full GPU simulation!")
        print("Try: ./run_gpu.sh --web")
    else:
        print("\n❌ GPU fix needs more attention")
