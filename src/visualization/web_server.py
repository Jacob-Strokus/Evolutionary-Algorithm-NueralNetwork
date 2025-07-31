#!/usr/bin/env python3
"""
Neural Ecosystem Web Server
Simple web interface for displaying the neural network ecosystem simulation
"""

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import json
import time
import threading

class EcosystemWebServer:
    """Simple web server for neural ecosystem visualization"""
    
    def __init__(self, ecosystem_canvas):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'neural_ecosystem_key'
        
        # Simple SocketIO setup
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.canvas = ecosystem_canvas
        self.running = False
        self.simulation_speed = 100  # milliseconds between updates
        self.step_count = 0
        
        self.setup_routes()
        self.setup_socket_events()
    
    def setup_routes(self):
        """Setup web routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string(self.get_html_template())
    
    def setup_socket_events(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print('✅ Client connected')
            emit('status', {'message': 'Connected to Neural Ecosystem'})
            # Send initial data
            ecosystem_data = self.get_ecosystem_data()
            emit('ecosystem_update', ecosystem_data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('❌ Client disconnected')
        
        @self.socketio.on('start_simulation')
        def handle_start():
            print('▶️ Starting simulation...')
            if not self.running:
                self.start_simulation_loop()
                emit('simulation_started', {'status': 'started'})
            else:
                emit('status', {'message': 'Simulation already running'})
        
        @self.socketio.on('stop_simulation')
        def handle_stop():
            print('⏹️ Stopping simulation...')
            self.running = False
            emit('simulation_stopped', {'status': 'stopped'})
        
        @self.socketio.on('set_speed')
        def handle_speed_change(data):
            self.simulation_speed = max(10, min(1000, data.get('speed', 100)))
            print(f'⚡ Speed set to {self.simulation_speed}ms')
            emit('status', {'message': f'Speed set to {self.simulation_speed}ms'})
        
        @self.socketio.on('inspect_agent')
        def handle_agent_inspection(data):
            agent_id = data.get('agent_id')
            print(f'🔍 Inspecting agent {agent_id}')
            agent_data = self.get_agent_details(agent_id)
            if agent_data:
                emit('agent_details', agent_data)
            else:
                emit('status', {'message': f'Agent {agent_id} not found'})
        
        @self.socketio.on('update_agent')
        def handle_agent_update(data):
            """Handle real-time agent updates for open inspector"""
            agent_id = data.get('agent_id')
            if agent_id:
                agent_data = self.get_agent_details(agent_id)
                if agent_data:
                    emit('agent_update', agent_data)
                else:
                    emit('agent_not_found', {'agent_id': agent_id})
    
    def get_ecosystem_data(self):
        """Get current ecosystem state"""
        if hasattr(self.canvas, 'get_agent_data'):
            agent_data = self.canvas.get_agent_data()
        else:
            # Fallback for simple canvas
            agent_data = {
                'herbivores': {'x': [], 'y': [], 'fitness': [], 'ids': []},
                'carnivores': {'x': [], 'y': [], 'fitness': [], 'ids': []},
                'food': {'x': [], 'y': []}
            }
            
            # Extract data from environment
            if hasattr(self.canvas, 'env'):
                env = self.canvas.env
                
                # Process agents
                for agent in env.agents:
                    x, y = agent.position.x, agent.position.y
                    fitness = getattr(agent.brain, 'fitness_score', 0) if hasattr(agent, 'brain') else 0
                    # Check both 'id' and 'agent_id' attributes, plus fallback to Python id()
                    agent_identifier = getattr(agent, 'id', getattr(agent, 'agent_id', id(agent)))
                    
                    if hasattr(agent, 'species_type'):
                        from src.core.ecosystem import SpeciesType
                        if agent.species_type == SpeciesType.HERBIVORE:
                            agent_data['herbivores']['x'].append(x)
                            agent_data['herbivores']['y'].append(y)
                            agent_data['herbivores']['fitness'].append(fitness)
                            agent_data['herbivores']['ids'].append(agent_identifier)
                        else:
                            agent_data['carnivores']['x'].append(x)
                            agent_data['carnivores']['y'].append(y)
                            agent_data['carnivores']['fitness'].append(fitness)
                            agent_data['carnivores']['ids'].append(agent_identifier)
                
                # Process food
                for food in env.food_sources:
                    if food.is_available:
                        agent_data['food']['x'].append(food.position.x)
                        agent_data['food']['y'].append(food.position.y)
        
        return {
            'step': self.step_count,
            'ecosystem': agent_data,
            'stats': {
                'herbivores': len(agent_data['herbivores']['x']),
                'carnivores': len(agent_data['carnivores']['x']),
                'food': len(agent_data['food']['x'])
            }
        }
    
    def get_agent_details(self, agent_id):
        """Get detailed information about a specific agent"""
        if not hasattr(self.canvas, 'env'):
            return None
        
        # Find the agent by ID
        target_agent = None
        for agent in self.canvas.env.agents:
            # Check both 'id' and 'agent_id' attributes, plus fallback to Python id()
            agent_identifier = getattr(agent, 'id', getattr(agent, 'agent_id', id(agent)))
            if agent_identifier == agent_id:
                target_agent = agent
                break
        
        if not target_agent:
            return None
        
        # Extract basic agent information
        from src.core.ecosystem import SpeciesType
        agent_identifier = getattr(target_agent, 'id', getattr(target_agent, 'agent_id', id(target_agent)))
        agent_details = {
            'id': agent_identifier,
            'species': 'herbivore' if target_agent.species_type == SpeciesType.HERBIVORE else 'carnivore',
            'position': {'x': target_agent.position.x, 'y': target_agent.position.y},
            'energy': target_agent.energy,
            'age': getattr(target_agent, 'age', 0),
            'generation': getattr(target_agent, 'generation', 0),
            'fitness': getattr(target_agent.brain, 'fitness_score', 0) if hasattr(target_agent, 'brain') else 0
        }
        
        # Extract neural network information if available
        if hasattr(target_agent, 'brain') and target_agent.brain is not None:
            nn = target_agent.brain  # The brain IS the neural network
            
            # Get current sensory inputs
            if hasattr(target_agent.brain, 'sensor_system'):
                try:
                    sensory_inputs = target_agent.brain.sensor_system.get_sensory_inputs(
                        target_agent, self.canvas.env
                    )
                except Exception as e:
                    print(f"Error getting sensory inputs: {e}")
                    sensory_inputs = [0] * 10  # Default fallback for 10-input network
            else:
                # Use the SensorSystem directly
                try:
                    from src.neural.neural_network import SensorSystem
                    sensory_inputs = SensorSystem.get_sensory_inputs(target_agent, self.canvas.env)
                except Exception as e:
                    print(f"Error getting sensory inputs: {e}")
                    sensory_inputs = [0] * 10  # Default fallback for 10-input network
            
            # Get network weights and structure
            try:
                # Get current network output
                import numpy as np
                current_output = nn.forward(np.array(sensory_inputs))
                
                # Calculate hidden layer activations for visualization
                inputs_array = np.array(sensory_inputs)
                hidden_raw = np.dot(inputs_array, nn.weights_input_hidden) + nn.bias_hidden
                hidden_activations = nn.tanh(hidden_raw)  # Apply activation function
                
                agent_details['neural_network'] = {
                    'input_size': getattr(nn.config, 'input_size', 8) if hasattr(nn, 'config') else 8,
                    'hidden_size': getattr(nn.config, 'hidden_size', 12) if hasattr(nn, 'config') else 12,
                    'output_size': getattr(nn.config, 'output_size', 4) if hasattr(nn, 'config') else 4,
                    'current_inputs': sensory_inputs,
                    'hidden_activations': hidden_activations.tolist() if hasattr(hidden_activations, 'tolist') else list(hidden_activations),
                    'weights_input_hidden': nn.weights_input_hidden.tolist() if hasattr(nn, 'weights_input_hidden') else [],
                    'weights_hidden_output': nn.weights_hidden_output.tolist() if hasattr(nn, 'weights_hidden_output') else [],
                    'bias_hidden': nn.bias_hidden.tolist() if hasattr(nn, 'bias_hidden') else [],
                    'bias_output': nn.bias_output.tolist() if hasattr(nn, 'bias_output') else [],
                    'input_labels': [
                        'Energy Level',
                        'Age Factor',
                        'Food Distance', 
                        'Food Angle',
                        'Threat/Prey Distance',
                        'Threat/Prey Angle',
                        'Population Density',
                        'Can Reproduce',
                        'X Boundary Distance',
                        'Y Boundary Distance'
                    ],
                    'output_labels': [
                        'Move X',
                        'Move Y',
                        'Reproduce',
                        'Intensity'
                    ],
                    'current_outputs': current_output.tolist() if hasattr(current_output, 'tolist') else list(current_output)
                }
            except Exception as e:
                print(f"Error processing neural network: {e}")
                agent_details['neural_network'] = None
        else:
            print(f"Agent {agent_identifier} has no brain attribute")
            agent_details['neural_network'] = None
        
        return agent_details
    
    def start_simulation_loop(self):
        """Start the simulation loop in a separate thread"""
        self.running = True
        
        def simulation_thread():
            while self.running:
                try:
                    # Update ecosystem
                    if hasattr(self.canvas, 'update_step'):
                        self.canvas.update_step()
                    elif hasattr(self.canvas, 'env'):
                        # Check if environment has step() method (common for neural environments)
                        if hasattr(self.canvas.env, 'step'):
                            self.canvas.env.step()
                        elif hasattr(self.canvas.env, 'update'):
                            self.canvas.env.update()
                        else:
                            print("❌ Environment has no step() or update() method")
                            self.running = False
                            break
                    
                    self.step_count += 1
                    
                    # Get current state and broadcast
                    ecosystem_data = self.get_ecosystem_data()
                    self.socketio.emit('ecosystem_update', ecosystem_data)
                    
                    # Sleep based on simulation speed
                    time.sleep(self.simulation_speed / 1000.0)
                    
                except Exception as e:
                    print(f"❌ Error in simulation loop: {e}")
                    self.running = False
                    break
        
        # Start simulation in background thread
        thread = threading.Thread(target=simulation_thread)
        thread.daemon = True
        thread.start()
    
    def get_html_template(self):
        """Return the HTML template for the web interface"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Ecosystem Simulation</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.0.0/d3.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .controls {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        
        button:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }
        
        #stopBtn {
            background: linear-gradient(45deg, #f44336, #d32f2f);
        }
        
        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        input[type="range"] {
            width: 150px;
        }
        
        .simulation-area {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            color: #333;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .canvas-container {
            text-align: center;
            margin-bottom: 20px;
        }
        
        #ecosystem-canvas {
            border: 2px solid #ddd;
            border-radius: 10px;
            background: #f9f9f9;
            max-width: 100%;
            height: auto;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        
        .connected {
            background: #4CAF50;
            color: white;
        }
        
        .disconnected {
            background: #f44336;
            color: white;
        }
        
        .log {
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            height: 150px;
            overflow-y: scroll;
            font-family: monospace;
            font-size: 12px;
            margin-top: 20px;
        }
        
        /* Agent Inspection Modal */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
        }
        
        .modal-content {
            background-color: white;
            margin: 2% auto;
            padding: 20px;
            border-radius: 15px;
            width: 90%;
            max-width: 1000px;
            max-height: 90vh;
            overflow-y: auto;
            color: #333;
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: #000;
        }
        
        .agent-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .info-card {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .neural-network-container {
            margin-top: 20px;
        }
        
        .neural-inputs, .neural-outputs {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }
        
        .input-value, .output-value {
            background: #e8f4f8;
            padding: 8px;
            border-radius: 5px;
            text-align: center;
            font-size: 12px;
            transition: all 0.3s ease;
        }
        
        .input-value.active {
            background: #4CAF50;
            color: white;
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
        }
        
        .output-value.active {
            background: #f44336;
            color: white;
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(244, 67, 54, 0.3);
        }
        
        .agent-info span {
            transition: all 0.2s ease;
        }
        
        .agent-info span.updated {
            background-color: #4CAF50;
            color: white;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        #neural-network-svg {
            width: 100%;
            height: 400px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: #fafafa;
        }
        
        .neural-node-active {
            animation: neuralPulse 1s ease-in-out infinite alternate;
        }
        
        @keyframes neuralPulse {
            from { transform: scale(1); }
            to { transform: scale(1.1); }
        }
        
        .connection-active {
            animation: connectionFlow 2s linear infinite;
        }
        
        @keyframes connectionFlow {
            0% { stroke-dasharray: 0 10; }
            100% { stroke-dasharray: 10 0; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Neural Ecosystem Simulation</h1>
            <p>Real-time visualization of neural network agents evolving in an ecosystem</p>
        </div>
        
        <div id="connection-status" class="connection-status disconnected">
            Disconnected
        </div>
        
        <div class="controls">
            <button id="startBtn">▶️ Start Simulation</button>
            <button id="stopBtn">⏹️ Stop Simulation</button>
            
            <div class="speed-control">
                <label>Speed:</label>
                <input type="range" id="speedSlider" min="10" max="500" value="100" step="10">
                <span id="speedValue">100ms</span>
            </div>
        </div>
        
        <div class="simulation-area">
            <div class="canvas-container">
                <canvas id="ecosystem-canvas" width="800" height="600"></canvas>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="step-count">0</div>
                    <div class="stat-label">Simulation Steps</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="herbivore-count">0</div>
                    <div class="stat-label">🦌 Herbivores</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="carnivore-count">0</div>
                    <div class="stat-label">🐺 Carnivores</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="food-count">0</div>
                    <div class="stat-label">🌱 Food Sources</div>
                </div>
            </div>
            
            <div class="log" id="log"></div>
        </div>
    </div>

    <!-- Agent Inspection Modal -->
    <div id="agentModal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <h2>🧠 Agent Inspector</h2>
            
            <div class="agent-info">
                <div class="info-card">
                    <h3>Basic Information</h3>
                    <p><strong>ID:</strong> <span id="agent-id">-</span></p>
                    <p><strong>Species:</strong> <span id="agent-species">-</span></p>
                    <p><strong>Energy:</strong> <span id="agent-energy">-</span></p>
                    <p><strong>Age:</strong> <span id="agent-age">-</span> steps</p>
                    <p><strong>Generation:</strong> <span id="agent-generation">-</span></p>
                    <p><strong>Fitness:</strong> <span id="agent-fitness">-</span></p>
                </div>
                
                <div class="info-card">
                    <h3>Position & Status</h3>
                    <p><strong>X:</strong> <span id="agent-x">-</span></p>
                    <p><strong>Y:</strong> <span id="agent-y">-</span></p>
                    <p><strong>Network Size:</strong> <span id="network-size">-</span></p>
                </div>
            </div>
            
            <div class="neural-network-container">
                <h3>🧠 Neural Network Visualization</h3>
                <svg id="neural-network-svg"></svg>
                
                <h4>Current Sensory Inputs</h4>
                <div class="neural-inputs" id="neural-inputs"></div>
                
                <h4>Current Neural Outputs</h4>
                <div class="neural-outputs" id="neural-outputs"></div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection
        const socket = io();
        let isConnected = false;
        let currentAgentData = null;
        
        // UI elements
        const canvas = document.getElementById('ecosystem-canvas');
        const ctx = canvas.getContext('2d');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const speedSlider = document.getElementById('speedSlider');
        const speedValue = document.getElementById('speedValue');
        const connectionStatus = document.getElementById('connection-status');
        const log = document.getElementById('log');
        const modal = document.getElementById('agentModal');
        const closeModal = document.getElementsByClassName('close')[0];
        
        // Socket events
        socket.on('connect', function() {
            isConnected = true;
            updateConnectionStatus(true);
            addLog('✅ Connected to server');
        });
        
        socket.on('disconnect', function() {
            isConnected = false;
            updateConnectionStatus(false);
            addLog('❌ Disconnected from server');
        });
        
        socket.on('ecosystem_update', function(data) {
            drawEcosystem(data.ecosystem);
            updateStats(data.stats, data.step);
            
            // Update agent inspector if open
            if (modal.style.display === 'block' && currentAgentData) {
                socket.emit('update_agent', { agent_id: currentAgentData.id });
            }
        });
        
        socket.on('simulation_started', function(data) {
            addLog('▶️ Simulation started');
            startBtn.disabled = true;
            stopBtn.disabled = false;
        });
        
        socket.on('simulation_stopped', function(data) {
            addLog('⏹️ Simulation stopped');
            startBtn.disabled = false;
            stopBtn.disabled = true;
        });
        
        socket.on('status', function(data) {
            addLog('ℹ️ ' + data.message);
        });
        
        socket.on('agent_details', function(data) {
            showAgentModal(data);
        });
        
        socket.on('agent_update', function(data) {
            updateAgentModal(data);
        });
        
        socket.on('agent_not_found', function(data) {
            addLog('⚠️ Agent ' + data.agent_id + ' no longer exists');
            modal.style.display = 'none';
        });
        
        // Button events
        startBtn.addEventListener('click', function() {
            if (isConnected) {
                socket.emit('start_simulation');
            } else {
                alert('Not connected to server');
            }
        });
        
        stopBtn.addEventListener('click', function() {
            if (isConnected) {
                socket.emit('stop_simulation');
            } else {
                alert('Not connected to server');
            }
        });
        
        speedSlider.addEventListener('input', function() {
            const speed = parseInt(speedSlider.value);
            speedValue.textContent = speed + 'ms';
            if (isConnected) {
                socket.emit('set_speed', {speed: speed});
            }
        });
        
        // Canvas click handler for agent inspection
        canvas.addEventListener('click', function(event) {
            if (!isConnected) return;
            
            const rect = canvas.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            const clickY = event.clientY - rect.top;
            
            // Convert to ecosystem coordinates
            const scaleX = 100 / canvas.width;
            const scaleY = 100 / canvas.height;
            const ecoX = clickX * scaleX;
            const ecoY = clickY * scaleY;
            
            // Find clicked agent
            const clickedAgent = findAgentAtPosition(ecoX, ecoY);
            if (clickedAgent) {
                socket.emit('inspect_agent', { agent_id: clickedAgent.id });
            }
        });
        
        // Modal close handlers
        closeModal.onclick = function() {
            modal.style.display = 'none';
        }
        
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
        
        // Helper functions
        function updateConnectionStatus(connected) {
            if (connected) {
                connectionStatus.textContent = 'Connected';
                connectionStatus.className = 'connection-status connected';
            } else {
                connectionStatus.textContent = 'Disconnected';
                connectionStatus.className = 'connection-status disconnected';
            }
        }
        
        function addLog(message) {
            const timestamp = new Date().toLocaleTimeString();
            log.innerHTML += timestamp + ': ' + message + '\\n';
            log.scrollTop = log.scrollHeight;
        }
        
        function updateStats(stats, step) {
            document.getElementById('step-count').textContent = step;
            document.getElementById('herbivore-count').textContent = stats.herbivores;
            document.getElementById('carnivore-count').textContent = stats.carnivores;
            document.getElementById('food-count').textContent = stats.food;
        }
        
        function drawEcosystem(ecosystem) {
            // Store ecosystem data for agent finding
            window.currentEcosystem = ecosystem;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Set up coordinate system (ecosystem is 100x100, canvas is 800x600)
            const scaleX = canvas.width / 100;
            const scaleY = canvas.height / 100;
            
            // Draw food sources
            ctx.fillStyle = '#8B4513';
            for (let i = 0; i < ecosystem.food.x.length; i++) {
                const x = ecosystem.food.x[i] * scaleX;
                const y = ecosystem.food.y[i] * scaleY;
                ctx.fillRect(x - 3, y - 3, 6, 6);
            }
            
            // Draw herbivores
            ctx.fillStyle = '#4CAF50';
            for (let i = 0; i < ecosystem.herbivores.x.length; i++) {
                const x = ecosystem.herbivores.x[i] * scaleX;
                const y = ecosystem.herbivores.y[i] * scaleY;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI);
                ctx.fill();
            }
            
            // Draw carnivores
            ctx.fillStyle = '#f44336';
            for (let i = 0; i < ecosystem.carnivores.x.length; i++) {
                const x = ecosystem.carnivores.x[i] * scaleX;
                const y = ecosystem.carnivores.y[i] * scaleY;
                ctx.beginPath();
                ctx.moveTo(x, y - 6);
                ctx.lineTo(x - 5, y + 4);
                ctx.lineTo(x + 5, y + 4);
                ctx.closePath();
                ctx.fill();
            }
        }
        
        function findAgentAtPosition(ecoX, ecoY) {
            if (!window.currentEcosystem) return null;
            
            const tolerance = 2; // Click tolerance in ecosystem units
            const ecosystem = window.currentEcosystem;
            
            // Check herbivores
            for (let i = 0; i < ecosystem.herbivores.x.length; i++) {
                const dx = ecosystem.herbivores.x[i] - ecoX;
                const dy = ecosystem.herbivores.y[i] - ecoY;
                if (Math.sqrt(dx * dx + dy * dy) < tolerance) {
                    return {
                        id: ecosystem.herbivores.ids[i],
                        species: 'herbivore'
                    };
                }
            }
            
            // Check carnivores
            for (let i = 0; i < ecosystem.carnivores.x.length; i++) {
                const dx = ecosystem.carnivores.x[i] - ecoX;
                const dy = ecosystem.carnivores.y[i] - ecoY;
                if (Math.sqrt(dx * dx + dy * dy) < tolerance) {
                    return {
                        id: ecosystem.carnivores.ids[i],
                        species: 'carnivore'
                    };
                }
            }
            
            return null;
        }
        
        function showAgentModal(agentData) {
            currentAgentData = agentData;
            updateAgentModalContent(agentData);
            modal.style.display = 'block';
        }
        
        function updateAgentModal(agentData) {
            currentAgentData = agentData;
            updateAgentModalContent(agentData);
        }
        
        function updateAgentModalContent(agentData) {
            // Update basic info with smooth transitions and highlight changes
            updateFieldWithHighlight('agent-id', agentData.id);
            updateFieldWithHighlight('agent-species', agentData.species);
            updateFieldWithHighlight('agent-energy', agentData.energy.toFixed(1));
            updateFieldWithHighlight('agent-age', agentData.age);
            updateFieldWithHighlight('agent-generation', agentData.generation);
            updateFieldWithHighlight('agent-fitness', agentData.fitness.toFixed(1));
            updateFieldWithHighlight('agent-x', agentData.position.x.toFixed(1));
            updateFieldWithHighlight('agent-y', agentData.position.y.toFixed(1));
            
            if (agentData.neural_network && agentData.neural_network !== null) {
                const nn = agentData.neural_network;
                updateFieldWithHighlight('network-size', `${nn.input_size}→${nn.hidden_size}→${nn.output_size}`);
                
                // Update neural inputs and outputs with real-time data
                updateNeuralInputs(nn);
                updateNeuralOutputs(nn);
                drawNeuralNetwork(nn);
                
                // Show neural network sections
                document.querySelector('.neural-network-container').style.display = 'block';
            } else {
                document.getElementById('network-size').textContent = 'No neural network data';
                document.querySelector('.neural-network-container').style.display = 'none';
                addLog('⚠️ Agent has no neural network data available');
            }
        }
        
        function updateFieldWithHighlight(elementId, newValue) {
            const element = document.getElementById(elementId);
            const oldValue = element.textContent;
            
            if (oldValue !== newValue.toString()) {
                element.textContent = newValue;
                element.classList.add('updated');
                setTimeout(() => {
                    element.classList.remove('updated');
                }, 500);
            }
        }
        
        function updateNeuralInputs(nn) {
            const container = document.getElementById('neural-inputs');
            container.innerHTML = '';
            
            if (!nn.current_inputs || !nn.input_labels) {
                container.innerHTML = '<div>No neural input data available</div>';
                return;
            }
            
            for (let i = 0; i < Math.min(nn.input_labels.length, nn.current_inputs.length); i++) {
                const div = document.createElement('div');
                div.className = 'input-value';
                const value = nn.current_inputs[i] || 0;
                if (Math.abs(value) > 0.3) {
                    div.className += ' active';
                }
                div.innerHTML = `<strong>${nn.input_labels[i]}</strong><br>${value.toFixed(3)}`;
                container.appendChild(div);
            }
        }
        
        function updateNeuralOutputs(nn) {
            const container = document.getElementById('neural-outputs');
            container.innerHTML = '';
            
            if (!nn.current_outputs || !nn.output_labels) {
                container.innerHTML = '<div>No neural output data available</div>';
                return;
            }
            
            // Find the strongest output for decision highlighting
            let maxValue = -Infinity;
            let maxIndex = -1;
            for (let i = 0; i < nn.current_outputs.length; i++) {
                if (Math.abs(nn.current_outputs[i]) > Math.abs(maxValue)) {
                    maxValue = nn.current_outputs[i];
                    maxIndex = i;
                }
            }
            
            for (let i = 0; i < Math.min(nn.output_labels.length, nn.current_outputs.length); i++) {
                const div = document.createElement('div');
                div.className = 'output-value';
                const value = nn.current_outputs[i] || 0;
                
                // Highlight active outputs and strongest decision
                if (Math.abs(value) > 0.3) {
                    div.className += ' active';
                }
                if (i === maxIndex && Math.abs(maxValue) > 0.1) {
                    div.className += ' strongest-decision';
                    div.style.border = '2px solid #FF9800';
                    div.style.fontWeight = 'bold';
                }
                
                div.innerHTML = `<strong>${nn.output_labels[i]}</strong><br>${value.toFixed(3)}`;
                container.appendChild(div);
            }
        }
        
        function drawNeuralNetwork(nn) {
            const svg = d3.select('#neural-network-svg');
            svg.selectAll('*').remove(); // Clear previous drawing
            
            const width = 800;
            const height = 400;
            svg.attr('width', width).attr('height', height);
            
            // Check if we have the necessary data
            if (!nn.weights_input_hidden || !nn.weights_hidden_output || 
                nn.weights_input_hidden.length === 0 || nn.weights_hidden_output.length === 0) {
                svg.append('text')
                    .attr('x', width / 2)
                    .attr('y', height / 2)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '16px')
                    .attr('fill', '#666')
                    .text('Neural network weights not available');
                return;
            }
            
            // Layer positions
            const inputX = 50;
            const hiddenX = width / 2;
            const outputX = width - 50;
            
            // Node positions
            const inputNodes = [];
            const hiddenNodes = [];
            const outputNodes = [];
            
            // Calculate input node positions
            for (let i = 0; i < nn.input_size; i++) {
                inputNodes.push({
                    x: inputX,
                    y: (height / (nn.input_size + 1)) * (i + 1),
                    value: nn.current_inputs[i] || 0,
                    label: nn.input_labels[i] || `Input ${i}`
                });
            }
            
            // Calculate hidden node positions
            for (let i = 0; i < nn.hidden_size; i++) {
                hiddenNodes.push({
                    x: hiddenX,
                    y: (height / (nn.hidden_size + 1)) * (i + 1),
                    value: nn.hidden_activations ? nn.hidden_activations[i] : 0 // Real hidden activations
                });
            }
            
            // Calculate output node positions
            for (let i = 0; i < nn.output_size; i++) {
                outputNodes.push({
                    x: outputX,
                    y: (height / (nn.output_size + 1)) * (i + 1),
                    value: nn.current_outputs[i] || 0,
                    label: nn.output_labels[i] || `Output ${i}`
                });
            }
            
            // Draw connections (input to hidden) with activation-based opacity
            for (let i = 0; i < inputNodes.length && i < nn.weights_input_hidden.length; i++) {
                for (let j = 0; j < hiddenNodes.length && j < nn.weights_input_hidden[i].length; j++) {
                    const weight = nn.weights_input_hidden[i][j];
                    const inputActivation = Math.abs(inputNodes[i].value);
                    const connectionStrength = Math.abs(weight) * inputActivation;
                    
                    svg.append('line')
                        .attr('x1', inputNodes[i].x)
                        .attr('y1', inputNodes[i].y)
                        .attr('x2', hiddenNodes[j].x)
                        .attr('y2', hiddenNodes[j].y)
                        .attr('stroke', weight > 0 ? '#4CAF50' : '#f44336')
                        .attr('stroke-width', Math.min(Math.abs(weight) * 3, 5))
                        .attr('opacity', Math.max(0.2, Math.min(connectionStrength * 2, 0.9)));
                }
            }
            
            // Draw connections (hidden to output) with activation-based opacity
            for (let i = 0; i < hiddenNodes.length && i < nn.weights_hidden_output.length; i++) {
                for (let j = 0; j < outputNodes.length && j < nn.weights_hidden_output[i].length; j++) {
                    const weight = nn.weights_hidden_output[i][j];
                    const hiddenActivation = Math.abs(hiddenNodes[i].value);
                    const connectionStrength = Math.abs(weight) * hiddenActivation;
                    
                    svg.append('line')
                        .attr('x1', hiddenNodes[i].x)
                        .attr('y1', hiddenNodes[i].y)
                        .attr('x2', outputNodes[j].x)
                        .attr('y2', outputNodes[j].y)
                        .attr('stroke', weight > 0 ? '#4CAF50' : '#f44336')
                        .attr('stroke-width', Math.min(Math.abs(weight) * 3, 5))
                        .attr('opacity', Math.max(0.2, Math.min(connectionStrength * 2, 0.9)));
                }
            }
            
            // Draw input nodes
            svg.selectAll('.input-node')
                .data(inputNodes)
                .enter()
                .append('circle')
                .attr('class', 'input-node')
                .attr('cx', d => d.x)
                .attr('cy', d => d.y)
                .attr('r', 15)
                .attr('fill', d => {
                    const intensity = Math.min(Math.abs(d.value), 1);
                    return d.value > 0 ? `rgba(76, 175, 80, ${0.3 + intensity * 0.7})` : 
                           d.value < 0 ? `rgba(244, 67, 54, ${0.3 + intensity * 0.7})` : '#e0e0e0';
                })
                .attr('stroke', '#333')
                .attr('stroke-width', 2);
            
            // Draw hidden nodes with activation-based coloring
            svg.selectAll('.hidden-node')
                .data(hiddenNodes)
                .enter()
                .append('circle')
                .attr('class', 'hidden-node')
                .attr('cx', d => d.x)
                .attr('cy', d => d.y)
                .attr('r', 12)
                .attr('fill', d => {
                    const intensity = Math.min(Math.abs(d.value), 1);
                    if (Math.abs(d.value) > 0.1) {
                        return d.value > 0 ? `rgba(76, 175, 80, ${0.3 + intensity * 0.7})` : 
                               `rgba(244, 67, 54, ${0.3 + intensity * 0.7})`;
                    } else {
                        return '#e0e0e0'; // Inactive neuron
                    }
                })
                .attr('stroke', '#333')
                .attr('stroke-width', 2);
            
            // Draw output nodes
            svg.selectAll('.output-node')
                .data(outputNodes)
                .enter()
                .append('circle')
                .attr('class', 'output-node')
                .attr('cx', d => d.x)
                .attr('cy', d => d.y)
                .attr('r', 15)
                .attr('fill', d => {
                    const intensity = Math.min(Math.abs(d.value), 1);
                    return d.value > 0 ? `rgba(76, 175, 80, ${0.3 + intensity * 0.7})` : 
                           d.value < 0 ? `rgba(244, 67, 54, ${0.3 + intensity * 0.7})` : '#e0e0e0';
                })
                .attr('stroke', '#333')
                .attr('stroke-width', 2);
            
            // Add labels for inputs
            svg.selectAll('.input-label')
                .data(inputNodes)
                .enter()
                .append('text')
                .attr('class', 'input-label')
                .attr('x', d => d.x - 25)
                .attr('y', d => d.y + 5)
                .attr('text-anchor', 'end')
                .attr('font-size', '9px')
                .attr('font-weight', 'bold')
                .text(d => d.label);
            
            // Add labels for outputs
            svg.selectAll('.output-label')
                .data(outputNodes)
                .enter()
                .append('text')
                .attr('class', 'output-label')
                .attr('x', d => d.x + 25)
                .attr('y', d => d.y + 5)
                .attr('text-anchor', 'start')
                .attr('font-size', '9px')
                .attr('font-weight', 'bold')
                .text(d => d.label);
            
            // Add layer labels
            svg.append('text')
                .attr('x', inputX)
                .attr('y', 20)
                .attr('text-anchor', 'middle')
                .attr('font-size', '14px')
                .attr('font-weight', 'bold')
                .text('Input Layer');
            
            svg.append('text')
                .attr('x', hiddenX)
                .attr('y', 20)
                .attr('text-anchor', 'middle')
                .attr('font-size', '14px')
                .attr('font-weight', 'bold')
                .text('Hidden Layer');
            
            svg.append('text')
                .attr('x', outputX)
                .attr('y', 20)
                .attr('text-anchor', 'middle')
                .attr('font-size', '14px')
                .attr('font-weight', 'bold')
                .text('Output Layer');
        }
        
        // Initialize
        addLog('🌐 Web interface loaded');
    </script>
</body>
</html>
        '''
    
    def start_server(self, host='localhost', port=5000, debug=False):
        """Start the web server"""
        print(f"🌐 Starting Neural Ecosystem Web Server...")
        print(f"📱 Open your browser to: http://{host}:{port}")
        print(f"⚡ Press Ctrl+C to stop the server")
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        except Exception as e:
            print(f"❌ Server error: {e}")

# Export the web server class
__all__ = ['EcosystemWebServer']
