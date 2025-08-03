#!/usr/bin/env python3
"""
Web Server Neural Network Bug Simulation
=========================================

Simulate the exact web server behavior to reproduce the neural network visualization bug
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main import Phase2NeuralEnvironment
import numpy as np
import traceback

def simulate_web_server_behavior():
    """Simulate the exact web server get_agent_details behavior"""
    print("🌐 WEB SERVER NEURAL NETWORK BUG SIMULATION")
    print("=" * 55)
    
    # Create environment
    env = Phase2NeuralEnvironment(width=200, height=200, use_neural_agents=True)
    
    # Run for various durations to test different agent states
    test_scenarios = [
        ("Fresh agents (0 steps)", 0),
        ("Early simulation (5 steps)", 5),
        ("Mid simulation (25 steps)", 25),
        ("Extended simulation (100 steps)", 100),
        ("Long simulation (200 steps)", 200)
    ]
    
    for scenario_name, steps in test_scenarios:
        print(f"\n🧪 TESTING: {scenario_name}")
        print("-" * 50)
        
        # Reset environment
        env = Phase2NeuralEnvironment(width=200, height=200, use_neural_agents=True)
        
        # Run simulation
        for step in range(steps):
            env.step()
        
        # Test agents in different states
        living_agents = [a for a in env.agents if a.is_alive]
        dead_agents = [a for a in env.agents if not a.is_alive]
        
        print(f"Living agents: {len(living_agents)}, Dead agents: {len(dead_agents)}")
        
        # Test living agents
        success_count = 0
        failure_count = 0
        
        for agent in living_agents[:5]:  # Test first 5 living agents
            try:
                agent_details = web_server_get_agent_details(agent, env)
                if agent_details and agent_details.get('neural_network'):
                    nn_data = agent_details['neural_network']
                    if (nn_data.get('weights_input_hidden') and 
                        nn_data.get('weights_hidden_output') and 
                        nn_data.get('current_inputs') and 
                        nn_data.get('current_outputs')):
                        success_count += 1
                    else:
                        failure_count += 1
                        print(f"    ❌ Agent {get_agent_id(agent)}: Missing required neural data")
                else:
                    failure_count += 1
                    print(f"    ❌ Agent {get_agent_id(agent)}: No neural network data")
            except Exception as e:
                failure_count += 1
                print(f"    💥 Agent {get_agent_id(agent)}: Exception - {e}")
        
        # Test dead agents (they might still have neural data)
        if dead_agents:
            print(f"  Testing dead agents...")
            for agent in dead_agents[:2]:
                try:
                    agent_details = web_server_get_agent_details(agent, env)
                    if agent_details and agent_details.get('neural_network'):
                        print(f"    Dead agent {get_agent_id(agent)}: Still has neural data")
                    else:
                        print(f"    Dead agent {get_agent_id(agent)}: No neural data")
                except Exception as e:
                    print(f"    Dead agent {get_agent_id(agent)}: Exception - {e}")
        
        print(f"  Results: {success_count} success, {failure_count} failures")
        
        if failure_count > 0:
            print(f"  ⚠️ Found {failure_count} failing agents in {scenario_name}")

def web_server_get_agent_details(agent, env):
    """Exact replica of the web server's get_agent_details method"""
    try:
        # This simulates the exact code from web_server.py
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
        
        # Extract neural network information if available
        if hasattr(agent, 'brain') and agent.brain is not None:
            nn = agent.brain  # The brain IS the neural network
            
            # Get current sensory inputs - Phase 2 compatible
            if hasattr(agent.brain, 'sensor_system'):
                try:
                    sensory_inputs = agent.brain.sensor_system.get_enhanced_sensory_inputs(
                        agent, env
                    )
                except Exception as e:
                    print(f"Error getting sensory inputs: {e}")
                    sensory_inputs = [0] * 25  # Phase 2 enhanced sensory inputs
            else:
                # Use the enhanced evolutionary sensor system for Phase 2
                try:
                    from src.neural.evolutionary_sensors import EvolutionarySensorSystem
                    sensory_inputs = EvolutionarySensorSystem.get_enhanced_sensory_inputs(agent, env)
                except Exception as e:
                    print(f"Error getting Phase 2 sensory inputs: {e}")
                    # Fallback to basic inputs - ensure we have enough inputs for the network
                    sensory_inputs = [
                        agent.energy / 100.0,  # Energy level
                        agent.age / 1000.0,    # Age factor
                        0.5, 0.0, 0.0,  # Food 1 (distance, x, y)
                        0.5, 0.0, 0.0,  # Food 2
                        0.5, 0.0, 0.0,  # Food 3
                        0.5, 0.0, 0.0,  # Threat 1 (distance, x, y)
                        0.5, 0.0, 0.0,  # Threat 2
                        0.5, 0.0, 0.0,  # Threat 3
                        0.5,  # Population density
                        1.0 if agent.can_reproduce() else 0.0,  # Can reproduce
                        agent.position.x / env.width,   # X boundary
                        agent.position.y / env.height,  # Y boundary
                        0.5,  # Exploration value
                        0.0,  # Social signal
                        0.0   # Extra padding
                    ]
            
            # Get network weights and structure
            try:
                # Ensure we have enough inputs for the network
                import numpy as np
                inputs_array = np.array(sensory_inputs)
                
                # For Phase 2 networks, ensure input size matches expectations
                expected_input_size = nn.weights_input_hidden.shape[0] if hasattr(nn, 'weights_input_hidden') else 26
                
                # Pad or truncate inputs to match network expectations
                if len(inputs_array) < expected_input_size:
                    # Pad with zeros
                    inputs_array = np.pad(inputs_array, (0, expected_input_size - len(inputs_array)))
                elif len(inputs_array) > expected_input_size:
                    # Truncate
                    inputs_array = inputs_array[:expected_input_size]
                
                # Get current network output using the network's own forward method
                current_output = nn.forward(inputs_array)
                
                # For Phase 2 EvolutionaryNeuralNetwork, get hidden activations from the network
                if hasattr(nn, 'hidden_state') and nn.hidden_state is not None:
                    # Use the network's internal hidden state
                    hidden_activations = nn.hidden_state
                else:
                    # Fallback: try to calculate manually if possible
                    try:
                        # For Phase 2 networks, we need to use enhanced inputs with memory context
                        if hasattr(nn, '_get_memory_context'):
                            memory_context = nn._get_memory_context()
                            # Only use first min_input_size inputs, then add memory
                            base_inputs = inputs_array[:getattr(nn.config, 'min_input_size', 20)]
                            enhanced_inputs = np.concatenate([base_inputs, memory_context])
                        else:
                            enhanced_inputs = inputs_array
                        
                        # Ensure input size matches network expectations
                        if enhanced_inputs.shape[0] != nn.weights_input_hidden.shape[0]:
                            target_size = nn.weights_input_hidden.shape[0]
                            if enhanced_inputs.shape[0] < target_size:
                                enhanced_inputs = np.pad(enhanced_inputs, (0, target_size - enhanced_inputs.shape[0]))
                            else:
                                enhanced_inputs = enhanced_inputs[:target_size]
                        
                        hidden_raw = np.dot(enhanced_inputs, nn.weights_input_hidden) + nn.bias_hidden
                        hidden_activations = np.tanh(hidden_raw)  # Apply activation function
                    except Exception as calc_error:
                        print(f"Could not calculate hidden activations: {calc_error}")
                        hidden_activations = np.zeros(nn.hidden_size if hasattr(nn, 'hidden_size') else 8)
                
                agent_details['neural_network'] = {
                    'input_size': getattr(nn.config, 'min_input_size', 20) if hasattr(nn, 'config') else 20,
                    'hidden_size': getattr(nn, 'hidden_size', 16) if hasattr(nn, 'hidden_size') else 16,
                    'output_size': getattr(nn.config, 'output_size', 6) if hasattr(nn, 'config') else 6,
                    'current_inputs': sensory_inputs[:20] if len(sensory_inputs) >= 20 else sensory_inputs,  # Show first 20 for display
                    'hidden_activations': hidden_activations.tolist() if hasattr(hidden_activations, 'tolist') else list(hidden_activations),
                    'weights_input_hidden': nn.weights_input_hidden.tolist() if hasattr(nn, 'weights_input_hidden') else [],
                    'weights_hidden_output': nn.weights_hidden_output.tolist() if hasattr(nn, 'weights_hidden_output') else [],
                    'bias_hidden': nn.bias_hidden.tolist() if hasattr(nn, 'bias_hidden') else [],
                    'bias_output': nn.bias_output.tolist() if hasattr(nn, 'bias_output') else [],
                    'input_labels': [
                        'Energy Level',
                        'Age Factor',
                        'Nearest Food Distance', 
                        'Nearest Food X',
                        'Nearest Food Y',
                        'Target 2 Distance',
                        'Target 2 X',
                        'Target 2 Y',
                        'Target 3 Distance',
                        'Target 3 X',
                        'Target 3 Y',
                        'Nearest Threat Distance',
                        'Nearest Threat X',
                        'Nearest Threat Y',
                        'Population Density',
                        'Can Reproduce',
                        'X Boundary Distance',
                        'Y Boundary Distance',
                        'Exploration Value',
                        'Social Signal'
                    ],
                    'output_labels': [
                        'Move X',
                        'Move Y',
                        'Reproduce',
                        'Intensity',
                        'Social Signal',
                        'Exploration Weight'
                    ],
                    'current_outputs': current_output.tolist() if hasattr(current_output, 'tolist') else list(current_output),
                    'has_memory': hasattr(nn, 'memory') and nn.memory is not None,
                    'has_recurrent': getattr(nn, 'has_recurrent', False),
                    'memory_size': getattr(nn.config, 'memory_size', 0) if hasattr(nn, 'config') else 0
                }
            except Exception as e:
                print(f"Error processing neural network: {e}")
                agent_details['neural_network'] = None
        else:
            print(f"Agent {agent_identifier} has no brain attribute")
            agent_details['neural_network'] = None
        
        return agent_details
    
    except Exception as e:
        print(f"Error in get_agent_details: {e}")
        return None

def get_agent_id(agent):
    """Helper to get agent ID consistently"""
    return getattr(agent, 'id', getattr(agent, 'agent_id', id(agent)))

if __name__ == "__main__":
    try:
        simulate_web_server_behavior()
    except Exception as e:
        print(f"\n💥 Simulation failed: {e}")
        traceback.print_exc()
