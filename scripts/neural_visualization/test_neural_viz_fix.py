#!/usr/bin/env python3
"""
Test Neural Network Visualization Fix
=====================================

Test the enhanced neural network visualization to ensure it handles edge cases properly
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main import Phase2NeuralEnvironment
import json

def test_enhanced_neural_viz():
    """Test the enhanced neural network visualization with edge cases"""
    print("🧪 TESTING ENHANCED NEURAL NETWORK VISUALIZATION")
    print("=" * 55)
    
    # Create environment and run for a bit
    env = Phase2NeuralEnvironment(width=200, height=200, use_neural_agents=True)
    
    # Run simulation for different durations to test various agent states
    for steps in [0, 10, 50, 100]:
        print(f"\n🔄 Testing after {steps} simulation steps")
        
        # Reset environment
        env = Phase2NeuralEnvironment(width=200, height=200, use_neural_agents=True)
        
        # Run simulation
        for step in range(steps):
            env.step()
        
        # Test web server data extraction
        test_agents = [a for a in env.agents if a.is_alive][:5]
        
        success_count = 0
        failure_count = 0
        
        for agent in test_agents:
            try:
                # Simulate web server get_agent_details
                agent_data = simulate_agent_details_extraction(agent, env)
                
                if validate_neural_data(agent_data):
                    success_count += 1
                    print(f"  ✅ Agent {get_agent_id(agent)}: Valid neural data")
                else:
                    failure_count += 1
                    print(f"  ❌ Agent {get_agent_id(agent)}: Invalid neural data")
                    
            except Exception as e:
                failure_count += 1
                print(f"  💥 Agent {get_agent_id(agent)}: Exception - {e}")
        
        print(f"  📊 Results: {success_count} success, {failure_count} failures")
    
    # Test edge cases
    print(f"\n🧪 TESTING EDGE CASES")
    print("-" * 30)
    
    test_edge_cases()

def simulate_agent_details_extraction(agent, env):
    """Simulate the enhanced web server agent details extraction"""
    from src.core.ecosystem import SpeciesType
    
    agent_identifier = getattr(agent, 'id', getattr(agent, 'agent_id', id(agent)))
    agent_details = {
        'id': agent_identifier,
        'species': 'herbivore' if agent.species_type == SpeciesType.HERBIVORE else 'carnivore',
        'position': {'x': agent.position.x, 'y': agent.position.y},
        'energy': agent.energy,
        'age': getattr(agent, 'age', 0),
        'generation': getattr(agent, 'generation', 0),
        'fitness': getattr(agent.brain, 'fitness_score', 0) if hasattr(agent, 'brain') else 0
    }
    
    # Extract neural network information
    if hasattr(agent, 'brain') and agent.brain is not None:
        nn = agent.brain
        
        # Get sensory inputs
        try:
            from src.neural.evolutionary_sensors import EvolutionarySensorSystem
            sensory_inputs = EvolutionarySensorSystem.get_enhanced_sensory_inputs(agent, env)
        except Exception as e:
            print(f"    Sensor error: {e}")
            sensory_inputs = create_fallback_inputs(agent, env)
        
        # Get network data
        try:
            import numpy as np
            inputs_array = np.array(sensory_inputs)
            
            # Ensure input size matches
            if hasattr(nn, 'weights_input_hidden'):
                expected_size = nn.weights_input_hidden.shape[0]
                if len(inputs_array) != expected_size:
                    if len(inputs_array) < expected_size:
                        inputs_array = np.pad(inputs_array, (0, expected_size - len(inputs_array)))
                    else:
                        inputs_array = inputs_array[:expected_size]
            
            # Get current output
            current_output = nn.forward(inputs_array)
            
            # Get hidden activations
            if hasattr(nn, 'hidden_state') and nn.hidden_state is not None:
                hidden_activations = nn.hidden_state
            else:
                # Calculate manually
                try:
                    hidden_raw = np.dot(inputs_array, nn.weights_input_hidden) + nn.bias_hidden
                    hidden_activations = np.tanh(hidden_raw)
                except:
                    hidden_activations = np.zeros(getattr(nn, 'hidden_size', 16))
            
            agent_details['neural_network'] = {
                'input_size': getattr(nn, 'input_size', len(inputs_array)),
                'hidden_size': getattr(nn, 'hidden_size', 16),
                'output_size': getattr(nn, 'output_size', len(current_output)),
                'current_inputs': inputs_array.tolist(),
                'hidden_activations': hidden_activations.tolist() if hasattr(hidden_activations, 'tolist') else list(hidden_activations),
                'weights_input_hidden': nn.weights_input_hidden.tolist() if hasattr(nn, 'weights_input_hidden') else [],
                'weights_hidden_output': nn.weights_hidden_output.tolist() if hasattr(nn, 'weights_hidden_output') else [],
                'current_outputs': current_output.tolist() if hasattr(current_output, 'tolist') else list(current_output),
                'input_labels': create_input_labels(),
                'output_labels': create_output_labels()
            }
            
        except Exception as e:
            print(f"    Neural network processing error: {e}")
            agent_details['neural_network'] = None
    else:
        agent_details['neural_network'] = None
    
    return agent_details

def validate_neural_data(agent_data):
    """Validate that neural network data meets the enhanced requirements"""
    if not agent_data or not agent_data.get('neural_network'):
        return False
    
    nn = agent_data['neural_network']
    
    # Check required fields exist and are valid
    required_fields = [
        'weights_input_hidden', 'weights_hidden_output',
        'current_inputs', 'current_outputs', 'hidden_activations'
    ]
    
    for field in required_fields:
        if field not in nn:
            print(f"    Missing field: {field}")
            return False
        
        if not isinstance(nn[field], list) or len(nn[field]) == 0:
            print(f"    Invalid or empty field: {field}")
            return False
    
    # Check dimensions make sense
    if nn['input_size'] <= 0 or nn['hidden_size'] <= 0 or nn['output_size'] <= 0:
        print(f"    Invalid network dimensions")
        return False
    
    # Check array dimensions match
    if len(nn['weights_input_hidden']) != nn['input_size']:
        print(f"    Input weight dimension mismatch")
        return False
    
    if len(nn['weights_hidden_output']) != nn['hidden_size']:
        print(f"    Hidden weight dimension mismatch")
        return False
    
    # Check for NaN/Infinity values
    def has_invalid_values(arr):
        """Check for NaN or infinity values in nested arrays"""
        if isinstance(arr, list):
            return any(has_invalid_values(item) for item in arr)
        import math
        return not isinstance(arr, (int, float)) or math.isnan(arr) or math.isinf(arr)
    
    for field in required_fields:
        if has_invalid_values(nn[field]):
            print(f"    NaN/Infinity values detected in {field}")
            return False
    
    return True

def test_edge_cases():
    """Test specific edge cases that might cause visualization failures"""
    edge_cases = [
        "Agent with corrupted weights",
        "Agent with mismatched dimensions", 
        "Agent with NaN values",
        "Agent with empty arrays"
    ]
    
    for case in edge_cases:
        print(f"  Testing: {case}")
        
        # Create mock neural network data with issues
        mock_nn_data = create_problematic_neural_data(case)
        
        # Test if enhanced validation catches the issue
        try:
            is_valid = validate_neural_data({'neural_network': mock_nn_data})
            if not is_valid:
                print(f"    ✅ Correctly detected invalid data")
            else:
                print(f"    ❌ Failed to detect invalid data")
        except Exception as e:
            print(f"    ✅ Exception caught: {e}")

def create_problematic_neural_data(case_type):
    """Create neural network data with specific problems"""
    if case_type == "Agent with corrupted weights":
        return {
            'input_size': 26, 'hidden_size': 16, 'output_size': 6,
            'weights_input_hidden': None,  # Corrupted!
            'weights_hidden_output': [[0.1, 0.2]] * 16,
            'current_inputs': [0.5] * 26,
            'current_outputs': [0.1] * 6,
            'hidden_activations': [0.3] * 16
        }
    elif case_type == "Agent with mismatched dimensions":
        return {
            'input_size': 26, 'hidden_size': 16, 'output_size': 6,
            'weights_input_hidden': [[0.1] * 16] * 20,  # Wrong size!
            'weights_hidden_output': [[0.1, 0.2]] * 16,
            'current_inputs': [0.5] * 26,
            'current_outputs': [0.1] * 6,
            'hidden_activations': [0.3] * 16
        }
    elif case_type == "Agent with NaN values":
        return {
            'input_size': 26, 'hidden_size': 16, 'output_size': 6,
            'weights_input_hidden': [[float('nan')] * 16] * 26,  # NaN values!
            'weights_hidden_output': [[0.1, 0.2]] * 16,
            'current_inputs': [0.5] * 26,
            'current_outputs': [0.1] * 6,
            'hidden_activations': [0.3] * 16
        }
    else:  # Empty arrays
        return {
            'input_size': 26, 'hidden_size': 16, 'output_size': 6,
            'weights_input_hidden': [],  # Empty!
            'weights_hidden_output': [],  # Empty!
            'current_inputs': [],
            'current_outputs': [],
            'hidden_activations': []
        }

def create_fallback_inputs(agent, env):
    """Create fallback sensory inputs"""
    return [agent.energy / 100.0, agent.age / 1000.0] + [0.5] * 24

def create_input_labels():
    """Create input labels"""
    return ['Energy', 'Age'] + [f'Input {i}' for i in range(2, 26)]

def create_output_labels():
    """Create output labels"""
    return ['Move X', 'Move Y', 'Reproduce', 'Intensity', 'Social', 'Exploration']

def get_agent_id(agent):
    """Get agent ID consistently"""
    return getattr(agent, 'id', getattr(agent, 'agent_id', id(agent)))

if __name__ == "__main__":
    try:
        test_enhanced_neural_viz()
        print(f"\n🎉 NEURAL NETWORK VISUALIZATION FIX TEST COMPLETE!")
        print("The enhanced validation should prevent visualization failures.")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
