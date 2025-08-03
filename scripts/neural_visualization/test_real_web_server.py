#!/usr/bin/env python3
"""
Test Neural Network Visualization Issues
========================================

Analyze the neural network visualization code to identify potential issues
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import time
import threading

def analyze_javascript_issues():
    """Analyze potential JavaScript issues in the neural network visualization"""
    print("🔍 ANALYZING JAVASCRIPT VISUALIZATION CODE")
    print("=" * 50)
    
    # Read the web server template
    try:
        from src.visualization.web_server import EcosystemWebServer
        server = EcosystemWebServer()
        html_template = server.get_html_template()
        
        # Look for potential issues in the neural network drawing code
        issues_found = []
        
        # Check for missing null checks
        if 'nn.weights_input_hidden' in html_template:
            if 'nn.weights_input_hidden && nn.weights_input_hidden.length > 0' not in html_template:
                issues_found.append("Missing null check for weights_input_hidden")
        
        # Check for missing error handling
        if 'drawNeuralNetwork' in html_template:
            if 'try {' not in html_template or 'catch' not in html_template:
                issues_found.append("Missing error handling in drawNeuralNetwork")
        
        # Check for array bounds checking
        if 'nn.weights_input_hidden[i][j]' in html_template:
            if 'i < nn.weights_input_hidden.length' not in html_template:
                issues_found.append("Potential array bounds issue")
        
        if issues_found:
            print("⚠️ Potential JavaScript issues found:")
            for issue in issues_found:
                print(f"  - {issue}")
        else:
            print("✅ No obvious JavaScript issues detected")
            
        # Look for the actual drawNeuralNetwork function
        if 'function drawNeuralNetwork' in html_template:
            print("✅ Found drawNeuralNetwork function")
            
            # Extract the function for analysis
            start_idx = html_template.find('function drawNeuralNetwork')
            if start_idx != -1:
                # Find the end of the function (simplified)
                brace_count = 0
                func_start = html_template.find('{', start_idx)
                func_end = func_start
                
                for i, char in enumerate(html_template[func_start:], func_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            func_end = i
                            break
                
                if func_end > func_start:
                    func_code = html_template[func_start:func_end+1]
                    
                    # Check for specific error conditions
                    if 'weights_input_hidden.length === 0' in func_code:
                        print("✅ Has empty weights check")
                    else:
                        print("⚠️ Missing empty weights check")
                    
                    if 'svg.append(\'text\')' in func_code and 'not available' in func_code:
                        print("✅ Has fallback error message")
                    else:
                        print("⚠️ Missing fallback error message")
                        
                    # Look for specific bug patterns
                    print(f"\n🔍 SPECIFIC BUG ANALYSIS:")
                    
                    # Check if there are issues with array access
                    if 'nn.weights_input_hidden[i]' in func_code and 'nn.weights_input_hidden[i].length' in func_code:
                        print("✅ Has proper 2D array access")
                    else:
                        print("⚠️ Potential 2D array access issue")
                    
                    # Check for proper null/undefined checks
                    checks = [
                        ('current_inputs', 'nn.current_inputs'),
                        ('hidden_activations', 'nn.hidden_activations'),
                        ('current_outputs', 'nn.current_outputs')
                    ]
                    
                    for check_name, check_var in checks:
                        if check_var in func_code:
                            if f'{check_var} || ' in func_code or f'!{check_var}' in func_code:
                                print(f"✅ Has null check for {check_name}")
                            else:
                                print(f"⚠️ Missing null check for {check_name}")
        else:
            print("❌ drawNeuralNetwork function not found")
            
    except Exception as e:
        print(f"❌ Failed to analyze JavaScript: {e}")
        import traceback
        traceback.print_exc()

def create_neural_viz_fix():
    """Create a fix for common neural network visualization issues"""
    print("\n🔧 CREATING NEURAL VISUALIZATION FIX")
    print("=" * 45)
    
    # Common issues and their fixes
    fixes = {
        "null_check": """
// Add null checks for neural network data
if (!nn || !nn.weights_input_hidden || !nn.weights_hidden_output || 
    !nn.current_inputs || !nn.current_outputs) {
    console.warn('Missing neural network data for agent');
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .text('Neural network data unavailable');
    return;
}""",
        
        "array_bounds": """
// Add array bounds checking
for (let i = 0; i < Math.min(inputNodes.length, nn.weights_input_hidden.length); i++) {
    for (let j = 0; j < Math.min(hiddenNodes.length, nn.weights_input_hidden[i].length); j++) {
        // Safe array access
    }
}""",
        
        "error_handling": """
// Add try-catch for neural network drawing
function drawNeuralNetwork(nn) {
    try {
        // ... existing code ...
    } catch (error) {
        console.error('Error drawing neural network:', error);
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', height / 2)
            .attr('text-anchor', 'middle')
            .text('Visualization error: ' + error.message);
    }
}"""
    }
    
    print("� Common fixes for neural visualization issues:")
    for fix_name, fix_code in fixes.items():
        print(f"\n{fix_name.upper()}:")
        print(fix_code)

if __name__ == "__main__":
    """Analyze potential JavaScript issues in the neural network visualization"""
    print("\n🔍 ANALYZING JAVASCRIPT VISUALIZATION CODE")
    print("=" * 50)
    
    # Read the web server template
    try:
        from src.visualization.web_server import EcosystemWebServer
        server = EcosystemWebServer()
        html_template = server.get_html_template()
        
        # Look for potential issues in the neural network drawing code
        issues_found = []
        
        # Check for missing null checks
        if 'nn.weights_input_hidden' in html_template:
            if 'nn.weights_input_hidden && nn.weights_input_hidden.length > 0' not in html_template:
                issues_found.append("Missing null check for weights_input_hidden")
        
        # Check for missing error handling
        if 'drawNeuralNetwork' in html_template:
            if 'try {' not in html_template or 'catch' not in html_template:
                issues_found.append("Missing error handling in drawNeuralNetwork")
        
        # Check for array bounds checking
        if 'nn.weights_input_hidden[i][j]' in html_template:
            if 'i < nn.weights_input_hidden.length' not in html_template:
                issues_found.append("Potential array bounds issue")
        
        if issues_found:
            print("⚠️ Potential JavaScript issues found:")
            for issue in issues_found:
                print(f"  - {issue}")
        else:
            print("✅ No obvious JavaScript issues detected")
            
        # Look for the actual drawNeuralNetwork function
        if 'function drawNeuralNetwork' in html_template:
            print("✅ Found drawNeuralNetwork function")
            
            # Extract the function for analysis
            start_idx = html_template.find('function drawNeuralNetwork')
            if start_idx != -1:
                # Find the end of the function (simplified)
                brace_count = 0
                func_start = html_template.find('{', start_idx)
                func_end = func_start
                
                for i, char in enumerate(html_template[func_start:], func_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            func_end = i
                            break
                
                if func_end > func_start:
                    func_code = html_template[func_start:func_end+1]
                    
                    # Check for specific error conditions
                    if 'weights_input_hidden.length === 0' in func_code:
                        print("✅ Has empty weights check")
                    else:
                        print("⚠️ Missing empty weights check")
                    
                    if 'svg.append(\'text\')' in func_code and 'not available' in func_code:
                        print("✅ Has fallback error message")
                    else:
                        print("⚠️ Missing fallback error message")
        else:
            print("❌ drawNeuralNetwork function not found")
            
    except Exception as e:
        print(f"❌ Failed to analyze JavaScript: {e}")

if __name__ == "__main__":
    # Analyze the JavaScript code for potential issues
    analyze_javascript_issues()
    
    # Create suggested fixes
    create_neural_viz_fix()
