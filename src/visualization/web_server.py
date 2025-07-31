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
                    agent_id = getattr(agent, 'agent_id', id(agent))
                    
                    if hasattr(agent, 'species_type'):
                        from src.core.ecosystem import SpeciesType
                        if agent.species_type == SpeciesType.HERBIVORE:
                            agent_data['herbivores']['x'].append(x)
                            agent_data['herbivores']['y'].append(y)
                            agent_data['herbivores']['fitness'].append(fitness)
                            agent_data['herbivores']['ids'].append(agent_id)
                        else:
                            agent_data['carnivores']['x'].append(x)
                            agent_data['carnivores']['y'].append(y)
                            agent_data['carnivores']['fitness'].append(fitness)
                            agent_data['carnivores']['ids'].append(agent_id)
                
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

    <script>
        // WebSocket connection
        const socket = io();
        let isConnected = false;
        
        // UI elements
        const canvas = document.getElementById('ecosystem-canvas');
        const ctx = canvas.getContext('2d');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const speedSlider = document.getElementById('speedSlider');
        const speedValue = document.getElementById('speedValue');
        const connectionStatus = document.getElementById('connection-status');
        const log = document.getElementById('log');
        
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
