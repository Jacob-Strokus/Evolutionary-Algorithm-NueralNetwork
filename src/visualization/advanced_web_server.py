"""
Advanced Real-time Web Server for Neural Ecosystem Visualization
================================================================

Modern web interface with:
- Real-time simulation display without page refresh
- Live metrics dashboard
- Neural diversity visualization
- Clickable agents for neural network inspection
- Interactive controls and statistics
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import time
import threading
import numpy as np
from datetime import datetime
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict, deque

class AdvancedEcosystemWebServer:
    """Advanced real-time web server for neural ecosystem visualization"""
    
    def __init__(self, ecosystem_canvas):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'neural_ecosystem_secret'
        # FIXED: Simplified SocketIO configuration without threading mode
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*"
        )
        self.canvas = ecosystem_canvas
        self.running = False
        self.clients_connected = 0
        self.simulation_speed = 100  # milliseconds between updates
        
        # Data tracking for metrics
        self.metrics_history = {
            'population': deque(maxlen=200),
            'energy': deque(maxlen=200),
            'generations': deque(maxlen=200),
            'fitness': deque(maxlen=200),
            'diversity': deque(maxlen=200)
        }
        
        self.setup_routes()
        self.setup_websocket_events()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return self.get_advanced_html_template()
        
        @self.app.route('/api/status')
        def status():
            return jsonify({
                'status': 'running' if self.running else 'stopped',
                'clients': self.clients_connected,
                'speed': self.simulation_speed,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/agent/<int:agent_id>')
        def get_agent_details(agent_id):
            """Get detailed information about a specific agent"""
            agent_data = self.get_agent_neural_data(agent_id)
            return jsonify(agent_data)
        
        @self.app.route('/api/neural_network/<int:agent_id>')
        def get_neural_network(agent_id):
            """Get neural network visualization for an agent"""
            network_viz = self.generate_neural_network_visualization(agent_id)
            return jsonify(network_viz)
    
    def setup_websocket_events(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.clients_connected += 1
            print(f"🔗 Client connected! Total clients: {self.clients_connected}")
            print(f"📡 Client ID: {request.sid}")
            emit('status', {'message': f'Connected! {self.clients_connected} clients online'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.clients_connected -= 1
            print(f"❌ Client disconnected. Total clients: {self.clients_connected}")
        
        @self.socketio.on('start_simulation')
        def handle_start():
            print("📥 Received start_simulation event")
            if not self.running:
                print("▶️ Starting simulation...")
                self.start_simulation_loop()
                emit('simulation_started', {'status': 'started'})
                print("📤 Sent simulation_started response")
            else:
                print("⚠️ Simulation already running")
                emit('status', {'message': 'Simulation already running'})
        
        @self.socketio.on('stop_simulation')
        def handle_stop():
            print("📥 Received stop_simulation event")
            print("⏹️ Stopping simulation...")
            self.running = False
            emit('simulation_stopped', {'status': 'stopped'})
            print("📤 Sent simulation_stopped response")
        
        @self.socketio.on('set_speed')
        def handle_speed_change(data):
            print(f"📥 Received set_speed event: {data}")
            old_speed = self.simulation_speed
            self.simulation_speed = max(10, min(1000, data.get('speed', 100)))
            print(f"⚡ Speed changed from {old_speed}ms to {self.simulation_speed}ms")
            emit('speed_changed', {'speed': self.simulation_speed})
        
        @self.socketio.on('inspect_agent')
        def handle_agent_inspection(data):
            print(f"📥 Received inspect_agent event: {data}")
            agent_id = data.get('agent_id')
            if agent_id is not None:
                print(f"🔍 Inspecting agent {agent_id}")
                agent_details = self.get_agent_neural_data(agent_id)
                emit('agent_details', agent_details)
                print("📤 Sent agent_details response")
            else:
                print("❌ No agent_id provided")
        
        @self.socketio.on_error_default
        def default_error_handler(e):
            print(f"❌ WebSocket error: {e}")
            import traceback
            traceback.print_exc()
    
    def start_simulation_loop(self):
        """Start the main simulation loop"""
        self.running = True
        
        def simulation_thread():
            step_count = 0
            while self.running and self.clients_connected > 0:
                try:
                    # Update simulation
                    self.canvas.update_step()
                    step_count += 1
                    
                    # Get current state
                    ecosystem_data = self.get_ecosystem_state()
                    
                    # Calculate metrics
                    metrics = self.calculate_real_time_metrics(ecosystem_data)
                    
                    # Update metrics history
                    self.update_metrics_history(metrics)
                    
                    # Emit data to all connected clients
                    self.socketio.emit('ecosystem_update', {
                        'step': step_count,
                        'ecosystem': ecosystem_data,
                        'metrics': metrics,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Emit metrics charts every 10 steps
                    if step_count % 10 == 0:
                        charts_data = self.generate_metrics_charts()
                        self.socketio.emit('metrics_update', charts_data)
                    
                    time.sleep(self.simulation_speed / 1000.0)
                    
                except Exception as e:
                    print(f"Simulation error: {e}")
                    break
            
            self.running = False
            self.socketio.emit('simulation_stopped', {'reason': 'completed'})
        
        thread = threading.Thread(target=simulation_thread)
        thread.daemon = True
        thread.start()
    
    def get_ecosystem_state(self):
        """Get current ecosystem state"""
        agent_data = self.canvas.get_agent_data()
        
        # Enhanced agent data with neural information
        enhanced_agents = {
            'herbivores': [],
            'carnivores': [],
            'food': agent_data['food']
        }
        
        # Process herbivores
        for i, (x, y, fitness, agent_id) in enumerate(zip(
            agent_data['herbivores']['x'],
            agent_data['herbivores']['y'], 
            agent_data['herbivores']['fitness'],
            agent_data['herbivores']['ids']
        )):
            enhanced_agents['herbivores'].append({
                'id': agent_id,
                'x': x,
                'y': y,
                'fitness': fitness,
                'energy': self.get_agent_energy(agent_id),
                'generation': self.get_agent_generation(agent_id),
                'age': self.get_agent_age(agent_id)
            })
        
        # Process carnivores
        for i, (x, y, fitness, agent_id) in enumerate(zip(
            agent_data['carnivores']['x'],
            agent_data['carnivores']['y'],
            agent_data['carnivores']['fitness'], 
            agent_data['carnivores']['ids']
        )):
            enhanced_agents['carnivores'].append({
                'id': agent_id,
                'x': x,
                'y': y,
                'fitness': fitness,
                'energy': self.get_agent_energy(agent_id),
                'generation': self.get_agent_generation(agent_id),
                'age': self.get_agent_age(agent_id)
            })
        
        return enhanced_agents
    
    def calculate_real_time_metrics(self, ecosystem_data):
        """Calculate real-time metrics"""
        herbivores = ecosystem_data['herbivores']
        carnivores = ecosystem_data['carnivores']
        
        total_agents = len(herbivores) + len(carnivores)
        
        if total_agents == 0:
            return {
                'population': {'total': 0, 'herbivores': 0, 'carnivores': 0},
                'energy': {'avg': 0, 'total': 0, 'herb_avg': 0, 'carn_avg': 0},
                'generations': {'avg': 0, 'max': 0, 'min': 0, 'diversity': 0},
                'fitness': {'avg': 0, 'max': 0, 'herb_avg': 0, 'carn_avg': 0},
                'neural_diversity': 0
            }
        
        # Population metrics
        population = {
            'total': total_agents,
            'herbivores': len(herbivores),
            'carnivores': len(carnivores)
        }
        
        # Energy metrics
        all_energies = [agent['energy'] for agent in herbivores + carnivores]
        herb_energies = [agent['energy'] for agent in herbivores]
        carn_energies = [agent['energy'] for agent in carnivores]
        
        energy = {
            'avg': np.mean(all_energies) if all_energies else 0,
            'total': sum(all_energies),
            'herb_avg': np.mean(herb_energies) if herb_energies else 0,
            'carn_avg': np.mean(carn_energies) if carn_energies else 0
        }
        
        # Generation metrics
        all_generations = [agent['generation'] for agent in herbivores + carnivores]
        unique_generations = set(all_generations)
        
        generations = {
            'avg': np.mean(all_generations) if all_generations else 0,
            'max': max(all_generations) if all_generations else 0,
            'min': min(all_generations) if all_generations else 0,
            'diversity': len(unique_generations)
        }
        
        # Fitness metrics
        all_fitness = [agent['fitness'] for agent in herbivores + carnivores]
        herb_fitness = [agent['fitness'] for agent in herbivores]
        carn_fitness = [agent['fitness'] for agent in carnivores]
        
        fitness = {
            'avg': np.mean(all_fitness) if all_fitness else 0,
            'max': max(all_fitness) if all_fitness else 0,
            'herb_avg': np.mean(herb_fitness) if herb_fitness else 0,
            'carn_avg': np.mean(carn_fitness) if carn_fitness else 0
        }
        
        # Neural diversity (simplified - based on generation diversity)
        neural_diversity = len(unique_generations) / max(1, len(all_generations)) if all_generations else 0
        
        return {
            'population': population,
            'energy': energy,
            'generations': generations,
            'fitness': fitness,
            'neural_diversity': neural_diversity
        }
    
    def update_metrics_history(self, metrics):
        """Update metrics history for charting"""
        timestamp = datetime.now()
        
        self.metrics_history['population'].append({
            'time': timestamp.isoformat(),
            'total': metrics['population']['total'],
            'herbivores': metrics['population']['herbivores'],
            'carnivores': metrics['population']['carnivores']
        })
        
        self.metrics_history['energy'].append({
            'time': timestamp.isoformat(),
            'avg': metrics['energy']['avg'],
            'herb_avg': metrics['energy']['herb_avg'],
            'carn_avg': metrics['energy']['carn_avg']
        })
        
        self.metrics_history['generations'].append({
            'time': timestamp.isoformat(),
            'avg': metrics['generations']['avg'],
            'max': metrics['generations']['max'],
            'diversity': metrics['generations']['diversity']
        })
        
        self.metrics_history['fitness'].append({
            'time': timestamp.isoformat(),
            'avg': metrics['fitness']['avg'],
            'herb_avg': metrics['fitness']['herb_avg'],
            'carn_avg': metrics['fitness']['carn_avg']
        })
        
        self.metrics_history['diversity'].append({
            'time': timestamp.isoformat(),
            'neural': metrics['neural_diversity']
        })
    
    def generate_metrics_charts(self):
        """Generate charts data for metrics"""
        return {
            'population_history': list(self.metrics_history['population']),
            'energy_history': list(self.metrics_history['energy']),
            'generation_history': list(self.metrics_history['generations']),
            'fitness_history': list(self.metrics_history['fitness']),
            'diversity_history': list(self.metrics_history['diversity'])
        }
    
    def get_agent_neural_data(self, agent_id):
        """Get detailed neural network data for an agent"""
        # Find agent in ecosystem
        agent = self.find_agent_by_id(agent_id)
        if not agent:
            return {'error': 'Agent not found'}
        
        try:
            neural_data = {
                'id': agent_id,
                'species': 'herbivore' if hasattr(agent, 'species_type') and agent.species_type.name == 'HERBIVORE' else 'carnivore',
                'energy': getattr(agent, 'energy', 0),
                'generation': getattr(agent, 'generation', 1),
                'age': getattr(agent, 'age', 0),
                'position': {
                    'x': agent.position.x if hasattr(agent, 'position') else 0,
                    'y': agent.position.y if hasattr(agent, 'position') else 0
                },
                'neural_network': {
                    'architecture': [8, 12, 4],  # Default architecture
                    'weights': [],
                    'biases': [],
                    'activations': []
                }
            }
            
            # Get neural network details if available
            if hasattr(agent, 'brain') or hasattr(agent, 'neural_network'):
                brain = getattr(agent, 'brain', None) or getattr(agent, 'neural_network', None)
                if brain:
                    # Get weights and biases
                    if hasattr(brain, 'weights'):
                        neural_data['neural_network']['weights'] = [w.tolist() if hasattr(w, 'tolist') else w for w in brain.weights]
                    if hasattr(brain, 'biases'):
                        neural_data['neural_network']['biases'] = [b.tolist() if hasattr(b, 'tolist') else b for b in brain.biases]
                    
                    # Get recent activations if available
                    if hasattr(brain, 'last_activations'):
                        neural_data['neural_network']['activations'] = brain.last_activations
            
            return neural_data
            
        except Exception as e:
            return {'error': f'Error getting neural data: {str(e)}'}
    
    def generate_neural_network_visualization(self, agent_id):
        """Generate neural network visualization"""
        agent_data = self.get_agent_neural_data(agent_id)
        
        if 'error' in agent_data:
            return agent_data
        
        try:
            # Create a simplified network visualization
            network = agent_data['neural_network']
            
            # Generate network structure data for D3.js visualization
            nodes = []
            links = []
            
            # Input layer
            for i in range(8):
                nodes.append({
                    'id': f'input_{i}',
                    'layer': 0,
                    'type': 'input',
                    'label': f'Input {i+1}',
                    'activation': network.get('activations', [{}])[0].get(f'input_{i}', 0)
                })
            
            # Hidden layer
            for i in range(12):
                nodes.append({
                    'id': f'hidden_{i}',
                    'layer': 1,
                    'type': 'hidden',
                    'label': f'Hidden {i+1}',
                    'activation': 0  # Placeholder
                })
            
            # Output layer
            output_labels = ['Move X', 'Move Y', 'Eat', 'Reproduce']
            for i in range(4):
                nodes.append({
                    'id': f'output_{i}',
                    'layer': 2,
                    'type': 'output',
                    'label': output_labels[i],
                    'activation': 0  # Placeholder
                })
            
            # Generate links (simplified)
            # Input to hidden
            for i in range(8):
                for j in range(12):
                    weight = 0
                    if network.get('weights') and len(network['weights']) > 0:
                        try:
                            weight = network['weights'][0][i][j] if len(network['weights'][0]) > i and len(network['weights'][0][i]) > j else 0
                        except:
                            weight = 0
                    
                    links.append({
                        'source': f'input_{i}',
                        'target': f'hidden_{j}',
                        'weight': weight
                    })
            
            # Hidden to output
            for i in range(12):
                for j in range(4):
                    weight = 0
                    if network.get('weights') and len(network['weights']) > 1:
                        try:
                            weight = network['weights'][1][i][j] if len(network['weights'][1]) > i and len(network['weights'][1][i]) > j else 0
                        except:
                            weight = 0
                    
                    links.append({
                        'source': f'hidden_{i}',
                        'target': f'output_{j}',
                        'weight': weight
                    })
            
            return {
                'agent_id': agent_id,
                'network_data': {
                    'nodes': nodes,
                    'links': links
                },
                'statistics': {
                    'total_weights': len(links),
                    'avg_weight': np.mean([link['weight'] for link in links]),
                    'max_weight': max([link['weight'] for link in links]) if links else 0,
                    'min_weight': min([link['weight'] for link in links]) if links else 0
                }
            }
            
        except Exception as e:
            return {'error': f'Error generating network visualization: {str(e)}'}
    
    def find_agent_by_id(self, agent_id):
        """Find agent by ID in the ecosystem"""
        try:
            if hasattr(self.canvas, 'env') and hasattr(self.canvas.env, 'agents'):
                for agent in self.canvas.env.agents:
                    if getattr(agent, 'agent_id', id(agent)) == agent_id:
                        return agent
            return None
        except:
            return None
    
    def get_agent_energy(self, agent_id):
        """Get agent energy"""
        agent = self.find_agent_by_id(agent_id)
        return getattr(agent, 'energy', 0) if agent else 0
    
    def get_agent_generation(self, agent_id):
        """Get agent generation"""
        agent = self.find_agent_by_id(agent_id)
        return getattr(agent, 'generation', 1) if agent else 1
    
    def get_agent_age(self, agent_id):
        """Get agent age"""
        agent = self.find_agent_by_id(agent_id)
        return getattr(agent, 'age', 0) if agent else 0
    
    def start_server(self, host='localhost', port=5000, debug=False):
        """Start the web server"""
        print(f"🌐 Starting Advanced Neural Ecosystem Web Server...")
        print(f"📱 Open your browser to: http://{host}:{port}")
        print(f"🧠 Features: Real-time simulation, neural inspection, live metrics")
        print(f"⚡ Press Ctrl+C to stop the server")
        print("\n🔧 Server Configuration:")
        print(f"   - Host: {host}")
        print(f"   - Port: {port}")
        print(f"   - Debug: {debug}")
        print(f"   - Transport: WebSocket + Polling")
        
        try:
            # FIXED: Simplified run configuration for better WebSocket connectivity
            self.socketio.run(
                self.app, 
                host=host, 
                port=port, 
                debug=debug
            )
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        except Exception as e:
            print(f"❌ Server error: {e}")
            import traceback
            traceback.print_exc()
    
    def get_advanced_html_template(self):
        """Return the advanced HTML template"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Neural Ecosystem Simulation</title>
    
    <!-- External libraries -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.5/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    
    <!-- Custom styles -->
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            overflow-x: hidden;
        }
        
        .header {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #f44336, #da190b);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(255,255,255,0.1);
            padding: 10px 20px;
            border-radius: 20px;
        }
        
        .speed-slider {
            width: 150px;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .simulation-panel {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .panel-title {
            font-size: 1.5em;
            margin-bottom: 15px;
            text-align: center;
            color: #FFD700;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        
        #ecosystem-canvas {
            width: 100%;
            height: 600px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 10px;
            background: #000;
            cursor: crosshair;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        
        .metric-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 5px;
        }
        
        .chart-container {
            position: relative;
            height: 200px;
            margin: 15px 0;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            padding: 10px;
        }
        
        .status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.8);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9em;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-light {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #f44336;
            animation: pulse 2s infinite;
        }
        
        .status-light.connected {
            background: #4CAF50;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        /* Modal styles for neural network inspection */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            backdrop-filter: blur(5px);
        }
        
        .modal-content {
            background: linear-gradient(135deg, #2a2a2a, #1a1a1a);
            margin: 5% auto;
            padding: 30px;
            border-radius: 15px;
            width: 90%;
            max-width: 1000px;
            color: white;
            border: 2px solid rgba(255,255,255,0.2);
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s;
        }
        
        .close:hover {
            color: #fff;
        }
        
        .neural-network-viz {
            width: 100%;
            height: 400px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 10px;
            background: #000;
        }
        
        .agent-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .info-card {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .info-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #FFD700;
        }
        
        /* Responsive design */
        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .dashboard {
                padding: 10px;
                gap: 15px;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <h1>🧠 Neural Ecosystem Simulation</h1>
        <p>Real-time evolution with clickable neural network inspection</p>
        
        <div class="controls">
            <button id="startBtn" class="btn btn-primary">▶️ Start Simulation</button>
            <button id="stopBtn" class="btn btn-danger">⏹️ Stop Simulation</button>
            
            <div class="speed-control">
                <label>Speed:</label>
                <input type="range" id="speedSlider" class="speed-slider" min="10" max="500" value="100">
                <span id="speedValue">100ms</span>
            </div>
        </div>
    </div>
    
    <!-- Main Dashboard -->
    <div class="dashboard">
        <!-- Simulation Display -->
        <div class="simulation-panel">
            <h2 class="panel-title">🌍 Live Ecosystem</h2>
            <canvas id="ecosystem-canvas"></canvas>
            <p style="text-align: center; margin-top: 10px; opacity: 0.8;">
                Click on agents to inspect their neural networks
            </p>
        </div>
        
        <!-- Metrics and Charts -->
        <div class="simulation-panel">
            <h2 class="panel-title">📊 Real-time Metrics</h2>
            
            <!-- Key metrics -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="totalPopulation">0</div>
                    <div class="metric-label">Total Population</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="avgEnergy">0</div>
                    <div class="metric-label">Avg Energy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="maxGeneration">0</div>
                    <div class="metric-label">Max Generation</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="neuralDiversity">0%</div>
                    <div class="metric-label">Neural Diversity</div>
                </div>
            </div>
            
            <!-- Population chart -->
            <div class="chart-container">
                <canvas id="populationChart"></canvas>
            </div>
            
            <!-- Energy chart -->
            <div class="chart-container">
                <canvas id="energyChart"></canvas>
            </div>
        </div>
    </div>
    
    <!-- Neural Network Inspection Modal -->
    <div id="neuralModal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <h2 id="modalTitle">🧠 Neural Network Inspector</h2>
            
            <div class="agent-info" id="agentInfo">
                <!-- Agent details will be populated here -->
            </div>
            
            <div class="neural-network-viz" id="networkViz">
                <!-- Neural network visualization will be rendered here -->
            </div>
        </div>
    </div>
    
    <!-- Status Bar -->
    <div class="status-bar">
        <div class="status-indicator">
            <div class="status-light" id="connectionStatus"></div>
            <span id="statusText">Disconnected</span>
        </div>
        <div>
            <span id="stepCount">Step: 0</span> | 
            <span id="clientCount">Clients: 0</span> | 
            <span id="timestamp">--</span>
        </div>
    </div>
    
    <!-- JavaScript -->
    <script>
        // Global variables
        let socket;
        let canvas, ctx;
        let currentStep = 0;
        let isConnected = false;
        let ecosystemData = null;
        let populationChart, energyChart;
        
        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initializeSocket();
            initializeCanvas();
            initializeCharts();
            initializeControls();
            initializeModal();
        });
        
        // Socket.IO initialization
        function initializeSocket() {
            // Create socket with polling-first configuration (more reliable)
            socket = io({
                transports: ['polling', 'websocket'],
                upgrade: true,
                timeout: 5000,
                forceNew: true
            });
            
            socket.on('connect', function() {
                isConnected = true;
                updateConnectionStatus(true);
                console.log('✅ Connected to server');
                console.log('Socket ID:', socket.id);
                console.log('Transport:', socket.io.engine.transport.name);
            });
            
            socket.on('disconnect', function(reason) {
                isConnected = false;
                updateConnectionStatus(false);
                console.log('❌ Disconnected from server:', reason);
            });
            
            socket.on('connect_error', function(error) {
                console.error('❌ Connection error:', error);
                updateConnectionStatus(false);
            });
            
            socket.on('ecosystem_update', function(data) {
                console.log('📊 Received ecosystem update:', data.step);
                currentStep = data.step;
                ecosystemData = data.ecosystem;
                updateSimulationDisplay(data);
                updateMetrics(data.metrics);
                updateStatusBar(data);
            });
            
            socket.on('metrics_update', function(data) {
                console.log('📈 Received metrics update');
                updateCharts(data);
            });
            
            socket.on('agent_details', function(data) {
                console.log('🧠 Received agent details');
                showAgentDetails(data);
            });
            
            socket.on('simulation_started', function() {
                console.log('▶️ Simulation started');
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
            });
            
            socket.on('simulation_stopped', function() {
                console.log('⏹️ Simulation stopped');
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            });
            
            socket.on('speed_changed', function(data) {
                console.log('⚡ Speed changed to:', data.speed);
            });
            
            socket.on('status', function(data) {
                console.log('📡 Status:', data.message);
            });
        }
        
        // Canvas initialization
        function initializeCanvas() {
            canvas = document.getElementById('ecosystem-canvas');
            ctx = canvas.getContext('2d');
            
            // Set canvas size
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            // Handle clicks for agent inspection
            canvas.addEventListener('click', function(event) {
                if (ecosystemData) {
                    const rect = canvas.getBoundingClientRect();
                    const x = (event.clientX - rect.left) * (100 / canvas.width);
                    const y = (event.clientY - rect.top) * (100 / canvas.height);
                    
                    // Find clicked agent
                    const clickedAgent = findAgentAt(x, y);
                    if (clickedAgent) {
                        socket.emit('inspect_agent', { agent_id: clickedAgent.id });
                    }
                }
            });
            
            // Draw initial empty canvas
            drawEmptyCanvas();
        }
        
        // Charts initialization
        function initializeCharts() {
            // Population chart
            const popCtx = document.getElementById('populationChart').getContext('2d');
            populationChart = new Chart(popCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Total',
                            data: [],
                            borderColor: '#FFD700',
                            backgroundColor: 'rgba(255, 215, 0, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Herbivores',
                            data: [],
                            borderColor: '#4CAF50',
                            backgroundColor: 'rgba(76, 175, 80, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Carnivores',
                            data: [],
                            borderColor: '#f44336',
                            backgroundColor: 'rgba(244, 67, 54, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Population Over Time',
                            color: 'white'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        x: {
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        }
                    }
                }
            });
            
            // Energy chart
            const energyCtx = document.getElementById('energyChart').getContext('2d');
            energyChart = new Chart(energyCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Average Energy',
                            data: [],
                            borderColor: '#FF9800',
                            backgroundColor: 'rgba(255, 152, 0, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Herbivore Avg',
                            data: [],
                            borderColor: '#4CAF50',
                            backgroundColor: 'rgba(76, 175, 80, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Carnivore Avg',
                            data: [],
                            borderColor: '#f44336',
                            backgroundColor: 'rgba(244, 67, 54, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Energy Levels Over Time',
                            color: 'white'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        x: {
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        }
                    }
                }
            });
        }
        
        // Controls initialization
        function initializeControls() {
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            
            startBtn.addEventListener('click', function() {
                console.log('🎯 Start button clicked');
                if (isConnected) {
                    console.log('📤 Emitting start_simulation event');
                    socket.emit('start_simulation');
                } else {
                    console.error('❌ Not connected to server');
                    alert('Not connected to server. Please refresh the page.');
                }
            });
            
            stopBtn.addEventListener('click', function() {
                console.log('🛑 Stop button clicked');
                if (isConnected) {
                    console.log('📤 Emitting stop_simulation event');
                    socket.emit('stop_simulation');
                } else {
                    console.error('❌ Not connected to server');
                    alert('Not connected to server. Please refresh the page.');
                }
            });
            
            const speedSlider = document.getElementById('speedSlider');
            const speedValue = document.getElementById('speedValue');
            
            speedSlider.addEventListener('input', function() {
                const speed = this.value;
                speedValue.textContent = speed + 'ms';
                console.log('⚡ Speed changed to:', speed);
                if (isConnected) {
                    socket.emit('set_speed', { speed: parseInt(speed) });
                } else {
                    console.error('❌ Not connected to server');
                }
            });
        }
        
        // Modal initialization
        function initializeModal() {
            const modal = document.getElementById('neuralModal');
            const closeBtn = document.querySelector('.close');
            
            closeBtn.addEventListener('click', function() {
                modal.style.display = 'none';
            });
            
            window.addEventListener('click', function(event) {
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            });
        }
        
        // Update simulation display
        function updateSimulationDisplay(data) {
            const ecosystem = data.ecosystem;
            
            // Clear canvas
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Scale factors
            const scaleX = canvas.width / 100;
            const scaleY = canvas.height / 100;
            
            // Draw food
            ctx.fillStyle = '#4CAF50';
            ecosystem.food.x.forEach((x, i) => {
                const y = ecosystem.food.y[i];
                ctx.beginPath();
                ctx.arc(x * scaleX, y * scaleY, 3, 0, 2 * Math.PI);
                ctx.fill();
            });
            
            // Draw herbivores
            ecosystem.herbivores.forEach(agent => {
                const size = 4 + (agent.energy / 50); // Size based on energy
                const alpha = 0.5 + (agent.generation / 20); // Transparency based on generation
                
                ctx.fillStyle = `rgba(76, 175, 80, ${Math.min(alpha, 1)})`;
                ctx.beginPath();
                ctx.arc(agent.x * scaleX, agent.y * scaleY, size, 0, 2 * Math.PI);
                ctx.fill();
                
                // Draw generation indicator
                ctx.fillStyle = '#FFD700';
                ctx.font = '10px Arial';
                ctx.fillText(agent.generation, agent.x * scaleX - 5, agent.y * scaleY - 8);
            });
            
            // Draw carnivores
            ecosystem.carnivores.forEach(agent => {
                const size = 5 + (agent.energy / 40); // Size based on energy
                const alpha = 0.5 + (agent.generation / 20); // Transparency based on generation
                
                ctx.fillStyle = `rgba(244, 67, 54, ${Math.min(alpha, 1)})`;
                ctx.beginPath();
                ctx.arc(agent.x * scaleX, agent.y * scaleY, size, 0, 2 * Math.PI);
                ctx.fill();
                
                // Draw generation indicator
                ctx.fillStyle = '#FFD700';
                ctx.font = '10px Arial';
                ctx.fillText(agent.generation, agent.x * scaleX - 5, agent.y * scaleY - 8);
            });
        }
        
        // Update metrics display
        function updateMetrics(metrics) {
            document.getElementById('totalPopulation').textContent = metrics.population.total;
            document.getElementById('avgEnergy').textContent = Math.round(metrics.energy.avg);
            document.getElementById('maxGeneration').textContent = metrics.generations.max;
            document.getElementById('neuralDiversity').textContent = Math.round(metrics.neural_diversity * 100) + '%';
        }
        
        // Update charts
        function updateCharts(data) {
            // Update population chart
            if (data.population_history && data.population_history.length > 0) {
                const latest = data.population_history.slice(-20); // Last 20 points
                const labels = latest.map((_, i) => i);
                
                populationChart.data.labels = labels;
                populationChart.data.datasets[0].data = latest.map(d => d.total);
                populationChart.data.datasets[1].data = latest.map(d => d.herbivores);
                populationChart.data.datasets[2].data = latest.map(d => d.carnivores);
                populationChart.update('none');
            }
            
            // Update energy chart
            if (data.energy_history && data.energy_history.length > 0) {
                const latest = data.energy_history.slice(-20); // Last 20 points
                const labels = latest.map((_, i) => i);
                
                energyChart.data.labels = labels;
                energyChart.data.datasets[0].data = latest.map(d => d.avg);
                energyChart.data.datasets[1].data = latest.map(d => d.herb_avg);
                energyChart.data.datasets[2].data = latest.map(d => d.carn_avg);
                energyChart.update('none');
            }
        }
        
        // Find agent at coordinates
        function findAgentAt(x, y) {
            if (!ecosystemData) return null;
            
            const tolerance = 5;
            
            // Check herbivores
            for (const agent of ecosystemData.herbivores) {
                if (Math.abs(agent.x - x) < tolerance && Math.abs(agent.y - y) < tolerance) {
                    return agent;
                }
            }
            
            // Check carnivores
            for (const agent of ecosystemData.carnivores) {
                if (Math.abs(agent.x - x) < tolerance && Math.abs(agent.y - y) < tolerance) {
                    return agent;
                }
            }
            
            return null;
        }
        
        // Show agent details in modal
        function showAgentDetails(agentData) {
            if (agentData.error) {
                alert('Error: ' + agentData.error);
                return;
            }
            
            // Update modal title
            document.getElementById('modalTitle').textContent = 
                `🧠 Neural Network: ${agentData.species} #${agentData.id}`;
            
            // Update agent info
            const agentInfo = document.getElementById('agentInfo');
            agentInfo.innerHTML = `
                <div class="info-card">
                    <div class="info-value">${agentData.species}</div>
                    <div>Species</div>
                </div>
                <div class="info-card">
                    <div class="info-value">${Math.round(agentData.energy)}</div>
                    <div>Energy</div>
                </div>
                <div class="info-card">
                    <div class="info-value">${agentData.generation}</div>
                    <div>Generation</div>
                </div>
                <div class="info-card">
                    <div class="info-value">${agentData.age}</div>
                    <div>Age</div>
                </div>
                <div class="info-card">
                    <div class="info-value">(${Math.round(agentData.position.x)}, ${Math.round(agentData.position.y)})</div>
                    <div>Position</div>
                </div>
            `;
            
            // Get and display neural network
            fetch(`/api/neural_network/${agentData.id}`)
                .then(response => response.json())
                .then(networkData => {
                    if (networkData.error) {
                        document.getElementById('networkViz').innerHTML = 
                            '<p style="text-align: center; padding: 50px;">Error loading neural network</p>';
                    } else {
                        renderNeuralNetwork(networkData);
                    }
                })
                .catch(error => {
                    document.getElementById('networkViz').innerHTML = 
                        '<p style="text-align: center; padding: 50px;">Error loading neural network</p>';
                });
            
            // Show modal
            document.getElementById('neuralModal').style.display = 'block';
        }
        
        // Render neural network visualization
        function renderNeuralNetwork(networkData) {
            const container = document.getElementById('networkViz');
            container.innerHTML = ''; // Clear previous content
            
            // Create SVG
            const svg = d3.select(container)
                .append('svg')
                .attr('width', '100%')
                .attr('height', '100%')
                .attr('viewBox', '0 0 800 400');
            
            const nodes = networkData.network_data.nodes;
            const links = networkData.network_data.links;
            
            // Position nodes
            const layerWidth = 250;
            const layerHeight = 350;
            
            nodes.forEach(node => {
                const layerNodes = nodes.filter(n => n.layer === node.layer);
                const nodeIndex = layerNodes.indexOf(node);
                
                node.x = 50 + node.layer * layerWidth;
                node.y = 50 + (nodeIndex * layerHeight / layerNodes.length);
            });
            
            // Draw links
            svg.selectAll('.link')
                .data(links)
                .enter()
                .append('line')
                .attr('class', 'link')
                .attr('x1', d => nodes.find(n => n.id === d.source).x)
                .attr('y1', d => nodes.find(n => n.id === d.source).y)
                .attr('x2', d => nodes.find(n => n.id === d.target).x)
                .attr('y2', d => nodes.find(n => n.id === d.target).y)
                .attr('stroke', d => d.weight > 0 ? '#4CAF50' : '#f44336')
                .attr('stroke-width', d => Math.abs(d.weight) * 2 + 0.5)
                .attr('opacity', 0.6);
            
            // Draw nodes
            svg.selectAll('.node')
                .data(nodes)
                .enter()
                .append('circle')
                .attr('class', 'node')
                .attr('cx', d => d.x)
                .attr('cy', d => d.y)
                .attr('r', 15)
                .attr('fill', d => {
                    switch(d.type) {
                        case 'input': return '#2196F3';
                        case 'hidden': return '#FF9800';
                        case 'output': return '#9C27B0';
                        default: return '#666';
                    }
                })
                .attr('stroke', '#fff')
                .attr('stroke-width', 2);
            
            // Add node labels
            svg.selectAll('.node-label')
                .data(nodes)
                .enter()
                .append('text')
                .attr('class', 'node-label')
                .attr('x', d => d.x)
                .attr('y', d => d.y - 25)
                .attr('text-anchor', 'middle')
                .attr('fill', '#fff')
                .attr('font-size', '10px')
                .text(d => d.label);
        }
        
        // Update connection status
        function updateConnectionStatus(connected) {
            const statusLight = document.getElementById('connectionStatus');
            const statusText = document.getElementById('statusText');
            
            if (connected) {
                statusLight.classList.add('connected');
                statusText.textContent = 'Connected';
            } else {
                statusLight.classList.remove('connected');
                statusText.textContent = 'Disconnected';
            }
        }
        
        // Update status bar
        function updateStatusBar(data) {
            document.getElementById('stepCount').textContent = `Step: ${data.step}`;
            document.getElementById('clientCount').textContent = `Clients: ${isConnected ? '1+' : '0'}`;
            document.getElementById('timestamp').textContent = new Date(data.timestamp).toLocaleTimeString();
        }
        
        // Draw empty canvas
        function drawEmptyCanvas() {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = '#666';
            ctx.font = '24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Click "Start Simulation" to begin', canvas.width/2, canvas.height/2);
        }
        
        // Handle window resize
        window.addEventListener('resize', function() {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            if (ecosystemData) {
                updateSimulationDisplay({ ecosystem: ecosystemData });
            } else {
                drawEmptyCanvas();
            }
        });
    </script>
</body>
</html>
        '''
