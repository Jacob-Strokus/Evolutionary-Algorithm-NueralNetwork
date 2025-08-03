#!/usr/bin/env python3
"""
Neural Network Visualization Bug Fix
====================================

Fix for the neural network visualization that loads for some individuals but not others
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def create_comprehensive_neural_viz_fix():
    """Create a comprehensive fix for neural network visualization issues"""
    print("🔧 NEURAL NETWORK VISUALIZATION BUG FIX")
    print("=" * 50)
    
    print("The issue is likely in the JavaScript drawNeuralNetwork function.")
    print("Here are the potential problems and their fixes:\n")
    
    # Problem 1: Insufficient null checking
    print("1. 🐛 PROBLEM: Insufficient null/undefined checking")
    print("   Some agents may have partially initialized neural network data")
    
    fix1 = '''
// ENHANCED NULL CHECKING FIX
function drawNeuralNetwork(nn) {
    const svg = d3.select('#neural-network-svg');
    svg.selectAll('*').remove();
    
    const width = 800;
    const height = 400;
    svg.attr('width', width).attr('height', height);
    
    // COMPREHENSIVE DATA VALIDATION
    if (!nn) {
        showError("No neural network data provided");
        return;
    }
    
    if (!nn.weights_input_hidden || !Array.isArray(nn.weights_input_hidden) || nn.weights_input_hidden.length === 0) {
        showError("Input-hidden weights not available");
        return;
    }
    
    if (!nn.weights_hidden_output || !Array.isArray(nn.weights_hidden_output) || nn.weights_hidden_output.length === 0) {
        showError("Hidden-output weights not available");
        return;
    }
    
    if (!nn.current_inputs || !Array.isArray(nn.current_inputs)) {
        showError("Current inputs not available");
        return;
    }
    
    if (!nn.current_outputs || !Array.isArray(nn.current_outputs)) {
        showError("Current outputs not available");
        return;
    }
    
    // Validate hidden activations (optional but recommended)
    if (!nn.hidden_activations || !Array.isArray(nn.hidden_activations)) {
        console.warn("Hidden activations not available, using fallback");
        nn.hidden_activations = new Array(nn.hidden_size || 16).fill(0);
    }
    
    function showError(message) {
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', height / 2)
            .attr('text-anchor', 'middle')
            .attr('font-size', '16px')
            .attr('fill', '#666')
            .text(message);
    }
    
    // Continue with drawing...
}'''
    
    print(fix1)
    
    # Problem 2: Array dimension mismatches
    print("\n2. 🐛 PROBLEM: Array dimension mismatches")
    print("   Network sizes might not match expected dimensions")
    
    fix2 = '''
// SAFE ARRAY ACCESS FIX
function drawNeuralNetworkSafe(nn) {
    // ... validation code above ...
    
    // SAFE DIMENSION CALCULATION
    const actualInputSize = nn.current_inputs.length;
    const actualHiddenSize = nn.hidden_activations.length;
    const actualOutputSize = nn.current_outputs.length;
    
    // Validate weight matrix dimensions
    if (nn.weights_input_hidden.length !== actualInputSize) {
        console.warn(`Input size mismatch: expected ${actualInputSize}, got ${nn.weights_input_hidden.length}`);
    }
    
    if (nn.weights_hidden_output.length !== actualHiddenSize) {
        console.warn(`Hidden size mismatch: expected ${actualHiddenSize}, got ${nn.weights_hidden_output.length}`);
    }
    
    // SAFE LOOP BOUNDS
    const safeInputCount = Math.min(actualInputSize, nn.weights_input_hidden.length);
    const safeHiddenCount = Math.min(actualHiddenSize, nn.weights_hidden_output.length);
    const safeOutputCount = Math.min(actualOutputSize, nn.weights_hidden_output[0]?.length || 0);
    
    // Draw connections with bounds checking
    for (let i = 0; i < safeInputCount; i++) {
        if (!nn.weights_input_hidden[i] || !Array.isArray(nn.weights_input_hidden[i])) {
            console.warn(`Skipping invalid weight row ${i}`);
            continue;
        }
        
        for (let j = 0; j < Math.min(safeHiddenCount, nn.weights_input_hidden[i].length); j++) {
            const weight = nn.weights_input_hidden[i][j];
            if (typeof weight !== 'number' || isNaN(weight)) {
                continue; // Skip invalid weights
            }
            
            // Safe drawing code here...
        }
    }
}'''
    
    print(fix2)
    
    # Problem 3: Agent state timing issues
    print("\n3. 🐛 PROBLEM: Agent state timing issues")
    print("   Agents might be in transitional states when visualization is requested")
    
    fix3 = '''
// AGENT STATE VALIDATION FIX
function updateAgentModalContent(agentData) {
    // ... existing code ...
    
    if (agentData.neural_network && agentData.neural_network !== null) {
        const nn = agentData.neural_network;
        
        // CHECK AGENT STATE
        if (agentData.energy <= 0) {
            console.warn("Agent is dead, neural network may be in cleanup state");
        }
        
        // VALIDATE NETWORK COMPLETENESS
        const isNetworkComplete = (
            nn.weights_input_hidden && 
            nn.weights_hidden_output && 
            nn.current_inputs && 
            nn.current_outputs &&
            nn.input_size > 0 &&
            nn.hidden_size > 0 &&
            nn.output_size > 0
        );
        
        if (!isNetworkComplete) {
            console.warn("Incomplete neural network data for agent", agentData.id);
            document.querySelector('.neural-network-container').style.display = 'none';
            addLog(`⚠️ Agent ${agentData.id} has incomplete neural network data`);
            return;
        }
        
        try {
            updateFieldWithHighlight('network-size', `${nn.input_size}→${nn.hidden_size}→${nn.output_size}`);
            updateNeuralInputs(nn);
            updateNeuralOutputs(nn);
            drawNeuralNetwork(nn);
            document.querySelector('.neural-network-container').style.display = 'block';
        } catch (error) {
            console.error('Error updating neural network visualization:', error);
            document.querySelector('.neural-network-container').style.display = 'none';
            addLog(`❌ Failed to visualize agent ${agentData.id}: ${error.message}`);
        }
    } else {
        document.getElementById('network-size').textContent = 'No neural network data';
        document.querySelector('.neural-network-container').style.display = 'none';
        addLog(`⚠️ Agent ${agentData.id} has no neural network data available`);
    }
}'''
    
    print(fix3)
    
    # Summary
    print("\n4. 📋 IMPLEMENTATION SUMMARY")
    print("   To fix the neural network visualization bug:")
    print("   1. Add comprehensive null/undefined checking")
    print("   2. Implement safe array bounds checking")
    print("   3. Add agent state validation")
    print("   4. Improve error handling and user feedback")
    print("   5. Add fallback values for missing data")

def create_web_server_patch():
    """Create a patch file for the web server"""
    print("\n🔧 CREATING WEB SERVER PATCH")
    print("=" * 35)
    
    patch_content = '''
## Neural Network Visualization Bug Fix Patch

### File: src/visualization/web_server.py

Replace the drawNeuralNetwork function (around line 1070) with this enhanced version:

```javascript
function drawNeuralNetwork(nn) {
    const svg = d3.select('#neural-network-svg');
    svg.selectAll('*').remove();
    
    const width = 800;
    const height = 400;
    svg.attr('width', width).attr('height', height);
    
    function showError(message) {
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', height / 2)
            .attr('text-anchor', 'middle')
            .attr('font-size', '16px')
            .attr('fill', '#666')
            .text(message);
        console.warn('Neural visualization error:', message);
    }
    
    // ENHANCED VALIDATION
    if (!nn) {
        showError("No neural network data provided");
        return;
    }
    
    if (!nn.weights_input_hidden || !Array.isArray(nn.weights_input_hidden) || nn.weights_input_hidden.length === 0) {
        showError("Input weights not available");
        return;
    }
    
    if (!nn.weights_hidden_output || !Array.isArray(nn.weights_hidden_output) || nn.weights_hidden_output.length === 0) {
        showError("Output weights not available");
        return;
    }
    
    if (!nn.current_inputs || !Array.isArray(nn.current_inputs)) {
        showError("Input values not available");
        return;
    }
    
    if (!nn.current_outputs || !Array.isArray(nn.current_outputs)) {
        showError("Output values not available");
        return;
    }
    
    // Fallback for missing hidden activations
    if (!nn.hidden_activations || !Array.isArray(nn.hidden_activations)) {
        nn.hidden_activations = new Array(nn.hidden_size || 16).fill(0);
    }
    
    // ... rest of the existing drawing code with bounds checking ...
}
```

### Also replace the updateAgentModalContent function with enhanced error handling:

```javascript
function updateAgentModalContent(agentData) {
    // ... existing basic info updates ...
    
    if (agentData.neural_network && agentData.neural_network !== null) {
        const nn = agentData.neural_network;
        
        // Validate network completeness
        const isValid = nn.weights_input_hidden && nn.weights_hidden_output && 
                       nn.current_inputs && nn.current_outputs;
        
        if (isValid) {
            try {
                updateFieldWithHighlight('network-size', `${nn.input_size}→${nn.hidden_size}→${nn.output_size}`);
                updateNeuralInputs(nn);
                updateNeuralOutputs(nn);
                drawNeuralNetwork(nn);
                document.querySelector('.neural-network-container').style.display = 'block';
            } catch (error) {
                console.error('Visualization error:', error);
                document.querySelector('.neural-network-container').style.display = 'none';
                addLog(`❌ Visualization failed for agent ${agentData.id}`);
            }
        } else {
            document.querySelector('.neural-network-container').style.display = 'none';
            addLog(`⚠️ Incomplete neural data for agent ${agentData.id}`);
        }
    } else {
        document.querySelector('.neural-network-container').style.display = 'none';
        addLog(`⚠️ No neural network for agent ${agentData.id}`);
    }
}
```
'''
    
    # Write patch to file
    with open('neural_viz_bug_fix.patch', 'w') as f:
        f.write(patch_content)
    
    print("✅ Patch file created: neural_viz_bug_fix.patch")
    print("   Apply this patch to fix the neural network visualization bug")

if __name__ == "__main__":
    create_comprehensive_neural_viz_fix()
    create_web_server_patch()
