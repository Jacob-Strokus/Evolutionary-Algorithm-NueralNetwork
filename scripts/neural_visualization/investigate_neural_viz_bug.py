#!/usr/bin/env python3
"""
Neural Network Visualization Bug Investigation
===============================================

Investigate why neural network visualization loads for some agents but not others
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main import Phase2NeuralEnvironment
from src.neural.evolutionary_agent import EvolutionaryNeuralAgent
from src.neural.neural_agents import NeuralAgent
from src.core.ecosystem import SpeciesType, Position
import numpy as np
import traceback
import json

def investigate_neural_visualization_bug():
    """Investigate the neural network visualization bug"""
    print("🔍 NEURAL NETWORK VISUALIZATION BUG INVESTIGATION")
    print("=" * 60)
    
    # Create test environment
    env = Phase2NeuralEnvironment(width=200, height=200, use_neural_agents=True)
    
    print(f"Created environment with {len(env.agents)} agents")
    
    # Run a few steps to establish agent states
    for step in range(10):
        env.step()
    
    # Test each agent to see which ones work and which don't
    working_agents = []
    broken_agents = []
    
    print(f"\n🧪 TESTING AGENT NEURAL NETWORK DATA EXTRACTION")
    print("-" * 50)
    
    for i, agent in enumerate(env.agents[:10]):  # Test first 10 agents
        if not agent.is_alive:
            continue
            
        print(f"\n🤖 Testing Agent {getattr(agent, 'id', getattr(agent, 'agent_id', 'unknown'))} ({agent.species_type.value})")
        
        try:
            # Simulate what the web server does to extract neural network data
            agent_data = extract_neural_network_data(agent, env)
            
            if agent_data and agent_data.get('neural_network'):
                nn_data = agent_data['neural_network']
                
                # Check for required fields
                required_fields = [
                    'weights_input_hidden', 'weights_hidden_output',
                    'current_inputs', 'current_outputs', 'hidden_activations'
                ]
                
                missing_fields = []
                for field in required_fields:
                    if field not in nn_data or not nn_data[field]:
                        missing_fields.append(field)
                
                if not missing_fields:
                    print(f"  ✅ SUCCESS: All neural data available")
                    print(f"    Network: {nn_data['input_size']}→{nn_data['hidden_size']}→{nn_data['output_size']}")
                    print(f"    Inputs: {len(nn_data['current_inputs'])} values")
                    print(f"    Hidden: {len(nn_data['hidden_activations'])} activations") 
                    print(f"    Outputs: {len(nn_data['current_outputs'])} values")
                    working_agents.append(agent)
                else:
                    print(f"  ⚠️ PARTIAL: Missing fields: {missing_fields}")
                    broken_agents.append((agent, missing_fields))
            else:
                print(f"  ❌ FAILED: No neural network data extracted")
                broken_agents.append((agent, ["all_data"]))
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            print(f"    Exception type: {type(e).__name__}")
            broken_agents.append((agent, [f"Exception: {e}"]))
    
    # Analyze results
    print(f"\n📊 ANALYSIS RESULTS")
    print("=" * 40)
    print(f"✅ Working agents: {len(working_agents)}")
    print(f"❌ Broken agents: {len(broken_agents)}")
    
    if working_agents:
        print(f"\n✅ WORKING AGENTS ANALYSIS:")
        for agent in working_agents[:3]:
            print(f"  Agent {getattr(agent, 'id', getattr(agent, 'agent_id', 'unknown'))}: {type(agent).__name__}")
            if hasattr(agent, 'brain'):
                print(f"    Brain type: {type(agent.brain).__name__}")
                print(f"    Has weights: {hasattr(agent.brain, 'weights_input_hidden')}")
                print(f"    Has forward: {hasattr(agent.brain, 'forward')}")
    
    if broken_agents:
        print(f"\n❌ BROKEN AGENTS ANALYSIS:")
        for agent, issues in broken_agents[:3]:
            print(f"  Agent {getattr(agent, 'id', getattr(agent, 'agent_id', 'unknown'))}: {type(agent).__name__}")
            if hasattr(agent, 'brain'):
                print(f"    Brain type: {type(agent.brain).__name__}")
                print(f"    Has weights: {hasattr(agent.brain, 'weights_input_hidden')}")
                print(f"    Issues: {issues}")
            else:
                print(f"    No brain attribute!")
    
    # Deep dive into one broken agent
    if broken_agents:
        print(f"\n🔬 DEEP DIVE INTO BROKEN AGENT")
        print("-" * 40)
        broken_agent, issues = broken_agents[0]
        deep_dive_agent_debug(broken_agent, env)
    
    # Test different agent types
    print(f"\n🧬 AGENT TYPE COMPARISON")
    print("-" * 30)
    
    agent_types = {}
    for agent in env.agents:
        agent_type = type(agent).__name__
        if agent_type not in agent_types:
            agent_types[agent_type] = []
        agent_types[agent_type].append(agent)
    
    for agent_type, agents in agent_types.items():
        working_count = sum(1 for a in agents if a in working_agents)
        broken_count = sum(1 for a, _ in broken_agents if a in agents)
        print(f"  {agent_type}: {working_count} working, {broken_count} broken")

def extract_neural_network_data(agent, env):
    """Extract neural network data like the web server does"""
    try:
        # Basic agent information
        agent_details = {
            'id': getattr(agent, 'id', id(agent)),
            'species': agent.species_type.value,
            'energy': agent.energy,
            'age': getattr(agent, 'age', 0),
            'generation': getattr(agent, 'generation', 0),
            'fitness': getattr(agent.brain, 'fitness_score', 0) if hasattr(agent, 'brain') else 0
        }
        
        # Extract neural network information
        if hasattr(agent, 'brain') and agent.brain is not None:
            nn = agent.brain
            
            # Get sensory inputs
            sensory_inputs = get_sensory_inputs_safe(agent, env)
            
            # Get network structure and weights
            try:
                inputs_array = np.array(sensory_inputs)
                
                # Ensure input size matches network
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
                hidden_activations = get_hidden_activations_safe(nn, inputs_array)
                
                agent_details['neural_network'] = {
                    'input_size': getattr(nn, 'input_size', len(inputs_array)),
                    'hidden_size': getattr(nn, 'hidden_size', 16),
                    'output_size': getattr(nn, 'output_size', len(current_output)),
                    'current_inputs': inputs_array.tolist(),
                    'hidden_activations': hidden_activations.tolist() if hasattr(hidden_activations, 'tolist') else list(hidden_activations),
                    'weights_input_hidden': nn.weights_input_hidden.tolist() if hasattr(nn, 'weights_input_hidden') else [],
                    'weights_hidden_output': nn.weights_hidden_output.tolist() if hasattr(nn, 'weights_hidden_output') else [],
                    'current_outputs': current_output.tolist() if hasattr(current_output, 'tolist') else list(current_output),
                    'input_labels': get_input_labels(),
                    'output_labels': get_output_labels()
                }
                
            except Exception as e:
                print(f"    Error processing neural network: {e}")
                agent_details['neural_network'] = None
        else:
            agent_details['neural_network'] = None
            
        return agent_details
        
    except Exception as e:
        print(f"    Error extracting agent data: {e}")
        return None

def get_sensory_inputs_safe(agent, env):
    """Safely get sensory inputs for any agent type"""
    try:
        # Try Phase 2 enhanced sensors first
        if hasattr(agent.brain, 'sensor_system'):
            return agent.brain.sensor_system.get_enhanced_sensory_inputs(agent, env)
        
        # Try evolutionary sensors
        try:
            from src.neural.evolutionary_sensors import EvolutionarySensorSystem
            return EvolutionarySensorSystem.get_enhanced_sensory_inputs(agent, env)
        except:
            pass
        
        # Try basic sensors
        try:
            from src.neural.neural_network import SensorSystem
            return SensorSystem.get_sensory_inputs(agent, env)
        except:
            pass
        
        # Fallback to basic inputs
        return create_fallback_inputs(agent, env)
        
    except Exception as e:
        print(f"    Error getting sensory inputs: {e}")
        return create_fallback_inputs(agent, env)

def get_hidden_activations_safe(nn, inputs_array):
    """Safely get hidden layer activations"""
    try:
        # Check if network has stored hidden state
        if hasattr(nn, 'hidden_state') and nn.hidden_state is not None:
            return nn.hidden_state
        
        # Try to calculate manually
        if hasattr(nn, 'weights_input_hidden') and hasattr(nn, 'bias_hidden'):
            hidden_raw = np.dot(inputs_array, nn.weights_input_hidden) + nn.bias_hidden
            return np.tanh(hidden_raw)
        
        # Fallback
        return np.zeros(getattr(nn, 'hidden_size', 16))
        
    except Exception as e:
        print(f"    Error getting hidden activations: {e}")
        return np.zeros(getattr(nn, 'hidden_size', 16))

def create_fallback_inputs(agent, env):
    """Create fallback sensory inputs"""
    return [
        agent.energy / 100.0,  # Energy level
        agent.age / 1000.0,    # Age
        0.5, 0.0, 0.0,  # Food data
        0.5, 0.0, 0.0,  # More food
        0.5, 0.0, 0.0,  # Even more food
        0.5, 0.0, 0.0,  # Threat data
        0.5, 0.0, 0.0,  # More threats
        0.5, 0.0, 0.0,  # Even more threats
        0.5,  # Population density
        1.0 if agent.can_reproduce() else 0.0,  # Reproduction
        agent.position.x / env.width,   # X boundary
        agent.position.y / env.height,  # Y boundary
        0.5,  # Exploration
        0.0,  # Social
        0.0,  # Extra
        0.0   # Padding
    ]

def get_input_labels():
    """Get standard input labels"""
    return [
        'Energy Level', 'Age Factor', 'Food 1 Dist', 'Food 1 X', 'Food 1 Y',
        'Food 2 Dist', 'Food 2 X', 'Food 2 Y', 'Food 3 Dist', 'Food 3 X', 'Food 3 Y',
        'Threat 1 Dist', 'Threat 1 X', 'Threat 1 Y', 'Threat 2 Dist', 'Threat 2 X', 'Threat 2 Y',
        'Threat 3 Dist', 'Threat 3 X', 'Threat 3 Y', 'Population', 'Can Reproduce',
        'X Boundary', 'Y Boundary', 'Exploration', 'Social'
    ]

def get_output_labels():
    """Get standard output labels"""
    return ['Move X', 'Move Y', 'Reproduce', 'Intensity', 'Social', 'Exploration']

def deep_dive_agent_debug(agent, env):
    """Deep dive debugging of a problematic agent"""
    print(f"Agent ID: {getattr(agent, 'id', getattr(agent, 'agent_id', 'unknown'))}")
    print(f"Agent Type: {type(agent).__name__}")
    print(f"Species: {agent.species_type.value}")
    print(f"Is Alive: {agent.is_alive}")
    print(f"Energy: {agent.energy}")
    
    if hasattr(agent, 'brain'):
        brain = agent.brain
        print(f"Brain Type: {type(brain).__name__}")
        print(f"Brain is None: {brain is None}")
        
        if brain is not None:
            print(f"Has weights_input_hidden: {hasattr(brain, 'weights_input_hidden')}")
            print(f"Has weights_hidden_output: {hasattr(brain, 'weights_hidden_output')}")
            print(f"Has forward method: {hasattr(brain, 'forward')}")
            print(f"Has config: {hasattr(brain, 'config')}")
            
            if hasattr(brain, 'weights_input_hidden'):
                print(f"Input weights shape: {brain.weights_input_hidden.shape}")
            if hasattr(brain, 'weights_hidden_output'):
                print(f"Output weights shape: {brain.weights_hidden_output.shape}")
                
            # Try to call forward method
            try:
                test_inputs = np.zeros(26)  # Try with 26 inputs
                output = brain.forward(test_inputs)
                print(f"Forward call successful: {len(output)} outputs")
            except Exception as e:
                print(f"Forward call failed: {e}")
    else:
        print("Agent has no brain attribute!")

if __name__ == "__main__":
    try:
        investigate_neural_visualization_bug()
    except Exception as e:
        print(f"\n💥 Investigation failed: {e}")
        traceback.print_exc()
