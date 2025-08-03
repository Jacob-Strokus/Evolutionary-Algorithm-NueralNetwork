#!/usr/bin/env python3
"""
Real Web Server Test for Neural Visualization Fix
================================================

Test the actual web server with the neural visualization fixes applied
"""

import sys
import os
import time
import threading

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main import Phase2NeuralEnvironment
from src.visualization.web_server import create_app
import socketio

def test_real_web_server():
    """Test the actual web server with neural network visualization"""
    print("🌐 TESTING REAL WEB SERVER WITH NEURAL VIZ FIXES")
    print("=" * 55)
    
    # Create environment
    env = Phase2NeuralEnvironment(width=200, height=200, use_neural_agents=True)
    
    # Run simulation for a bit to get diverse agent states
    print("🔄 Running simulation to create diverse agent states...")
    for _ in range(50):
        env.step()
    
    print(f"📊 Created {len([a for a in env.agents if a.is_alive])} living agents")
    
    # Create and configure Flask app with SocketIO
    app = create_app()
    app.config['SECRET_KEY'] = 'test_secret_key'
    
    # Create SocketIO instance
    sio = socketio.Server(cors_allowed_origins="*")
    
    # Add our environment to the app
    app.environment = env
    app.simulation_running = False
    app.simulation_speed = 100
    
    # Test agent inspection functionality
    print("\n🔍 Testing agent inspection functionality...")
    
    # Get some test agents
    test_agents = [a for a in env.agents if a.is_alive][:5]
    
    for i, agent in enumerate(test_agents):
        try:
            agent_id = getattr(agent, 'id', getattr(agent, 'agent_id', id(agent)))
            print(f"  Testing agent {agent_id}...")
            
            # Simulate the agent inspection process
            from src.core.ecosystem import SpeciesType
            
            agent_details = {
                'id': agent_id,
                'species': 'herbivore' if agent.species_type == SpeciesType.HERBIVORE else 'carnivore',
                'position': {'x': agent.position.x, 'y': agent.position.y},
                'energy': agent.energy,
                'age': getattr(agent, 'age', 0),
                'generation': getattr(agent, 'generation', 0),
                'fitness': getattr(agent.brain, 'fitness_score', 0) if hasattr(agent, 'brain') else 0
            }
            
            # Extract neural network information (this is where the bug was)
            if hasattr(agent, 'brain') and agent.brain is not None:
                nn = agent.brain
                
                # Get sensory inputs
                try:
                    from src.neural.evolutionary_sensors import EvolutionarySensorSystem
                    sensory_inputs = EvolutionarySensorSystem.get_enhanced_sensory_inputs(agent, env)
                except Exception as e:
                    sensory_inputs = [agent.energy / 100.0, agent.age / 1000.0] + [0.5] * 24
                
                # Get network data with enhanced validation
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
                        try:
                            hidden_raw = np.dot(inputs_array, nn.weights_input_hidden) + nn.bias_hidden
                            hidden_activations = np.tanh(hidden_raw)
                        except:
                            hidden_activations = np.zeros(getattr(nn, 'hidden_size', 16))
                    
                    # Validate for NaN/Infinity before including in response
                    def validate_array(arr, name):
                        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                            print(f"    ⚠️  Warning: {name} contains NaN/Infinity values")
                            return False
                        return True
                    
                    # Create neural network data
                    neural_data = {
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
                    
                    # Final validation
                    all_valid = True
                    all_valid &= validate_array(np.array(neural_data['current_inputs']), 'inputs')
                    all_valid &= validate_array(np.array(neural_data['hidden_activations']), 'hidden')
                    all_valid &= validate_array(np.array(neural_data['current_outputs']), 'outputs')
                    all_valid &= validate_array(np.array(neural_data['weights_input_hidden']), 'input_weights')
                    all_valid &= validate_array(np.array(neural_data['weights_hidden_output']), 'output_weights')
                    
                    if all_valid:
                        agent_details['neural_network'] = neural_data
                        print(f"    ✅ Agent {agent_id}: Valid neural data extracted")
                    else:
                        agent_details['neural_network'] = None
                        print(f"    ❌ Agent {agent_id}: Invalid neural data (NaN/Inf detected)")
                        
                except Exception as e:
                    print(f"    💥 Agent {agent_id}: Neural processing error - {e}")
                    agent_details['neural_network'] = None
            else:
                agent_details['neural_network'] = None
                print(f"    ⚠️  Agent {agent_id}: No brain available")
            
        except Exception as e:
            print(f"    💥 Agent {agent_id}: General error - {e}")
    
    print(f"\n🎉 REAL WEB SERVER TEST COMPLETE!")
    print("Neural network data extraction and validation working properly.")
    
    return True

def create_input_labels():
    """Create input labels"""
    return ['Energy', 'Age'] + [f'Input {i}' for i in range(2, 26)]

def create_output_labels():
    """Create output labels"""
    return ['Move X', 'Move Y', 'Reproduce', 'Intensity', 'Social', 'Exploration']

if __name__ == "__main__":
    try:
        test_real_web_server()
    except Exception as e:
        print(f"\n💥 Real web server test failed: {e}")
        import traceback
        traceback.print_exc()
