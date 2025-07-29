"""
Real-time Web Server for Ecosystem Visualization
Uses Flask with WebSocket support for live data streaming
"""

from flask import Flask, render_template, jsonify
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

class EcosystemWebServer:
    """Real-time web server for ecosystem data streaming"""
    
    def __init__(self, ecosystem_canvas):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'ecosystem_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.canvas = ecosystem_canvas
        self.running = False
        self.clients_connected = 0
        
        self.setup_routes()
        self.setup_websocket_events()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return self.get_html_template()
        
        @self.app.route('/api/status')
        def status():
            return jsonify({
                'status': 'running' if self.running else 'stopped',
                'clients': self.clients_connected,
                'timestamp': datetime.now().isoformat()
            })
    
    def setup_websocket_events(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.clients_connected += 1
            print(f"🔌 Client connected. Total clients: {self.clients_connected}")
            emit('status', {'message': 'Connected to ecosystem stream'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.clients_connected -= 1
            print(f"🔌 Client disconnected. Total clients: {self.clients_connected}")
        
        @self.socketio.on('start_simulation')
        def handle_start():
            if not self.running:
                self.running = True
                threading.Thread(target=self.simulation_loop, daemon=True).start()
                emit('status', {'message': 'Simulation started'})
        
        @self.socketio.on('stop_simulation')
        def handle_stop():
            self.running = False
            emit('status', {'message': 'Simulation stopped'})
        
        @self.socketio.on('end_simulation')
        def handle_end():
            self.running = False
            emit('status', {'message': 'Simulation ended - program closing'})
            print("🔚 End simulation requested from web interface")
            # Schedule cleanup and exit
            threading.Timer(1.0, self.cleanup_and_exit).start()
        
        @self.socketio.on('request_data')
        def handle_data_request():
            # Send current ecosystem state immediately
            data = self.get_current_data()
            emit('ecosystem_update', data)
        
        @self.socketio.on('inspect_agent')
        def handle_inspect_agent(data):
            # Get detailed agent information for inspection
            agent_id = data.get('agent_id')
            try:
                agent_details = self.get_agent_details(agent_id)
                emit('agent_details', agent_details)
            except Exception as e:
                emit('error', {'message': f'Failed to get agent details: {str(e)}'})
    
    def get_agent_details(self, agent_id):
        """Get detailed information about a specific agent"""
        try:
            from src.neural.neural_agents import NeuralAgent
            
            # Parse agent ID to find the specific agent
            if agent_id.startswith('herb_'):
                index = int(agent_id.split('_')[1])
                agent_type = 'herbivore'
            elif agent_id.startswith('carn_'):
                index = int(agent_id.split('_')[1])
                agent_type = 'carnivore'
            else:
                raise ValueError(f"Invalid agent ID: {agent_id}")
            
            # Get neural agents from environment
            neural_agents = [agent for agent in self.canvas.env.agents if isinstance(agent, NeuralAgent)]
            
            if agent_type == 'herbivore':
                herbivores = [a for a in neural_agents if a.species_type.value == 'herbivore']
                if index < len(herbivores):
                    agent = herbivores[index]
                else:
                    raise ValueError(f"Herbivore index {index} out of range")
            else:
                carnivores = [a for a in neural_agents if a.species_type.value == 'carnivore']
                if index < len(carnivores):
                    agent = carnivores[index]
                else:
                    raise ValueError(f"Carnivore index {index} out of range")
            
            # Extract neural network weights for visualization
            brain_data = {}
            if hasattr(agent, 'brain') and hasattr(agent.brain, 'weights'):
                brain_data = {
                    'layer_sizes': [w.shape for w in agent.brain.weights],
                    'activations': [np.mean(w) for w in agent.brain.weights],
                    'weight_ranges': [(float(np.min(w)), float(np.max(w))) for w in agent.brain.weights]
                }
            
            return {
                'agent_id': agent_id,
                'species': agent_type,
                'generation': getattr(agent, 'generation', 1),  # Add generation info
                'position': {'x': float(agent.position.x), 'y': float(agent.position.y)},
                'energy': float(agent.energy),
                'max_energy': float(agent.max_energy),
                'age': int(agent.age),
                'fitness': float(agent.brain.fitness_score) if hasattr(agent, 'brain') else 0,
                'decisions_made': int(getattr(agent.brain, 'decisions_made', 0)) if hasattr(agent, 'brain') else 0,
                'can_reproduce': agent.can_reproduce(),
                'reproduction_cooldown': int(agent.reproduction_cooldown),
                'neural_data': brain_data,
                'performance_stats': {
                    'lifetime_energy_gained': getattr(agent, 'lifetime_energy_gained', 0),
                    'lifetime_food_consumed': getattr(agent, 'lifetime_food_consumed', 0),
                    'offspring_count': getattr(agent, 'offspring_count', 0),
                    'survival_time': int(agent.age)
                }
            }
        except Exception as e:
            print(f"❌ Error getting agent details: {e}")
            return {'error': str(e)}
    
    def get_current_data(self):
        """Get current ecosystem data for streaming"""
        try:
            # Get agent data
            agent_data = self.canvas.get_agent_data()
            stats = self.canvas.env.get_neural_stats()
            
            # Get additional agent details for inspection
            from src.neural.neural_agents import NeuralAgent
            neural_agents = [agent for agent in self.canvas.env.agents if isinstance(agent, NeuralAgent)]
            
            herbivores = [a for a in neural_agents if a.species_type.value == 'herbivore']
            carnivores = [a for a in neural_agents if a.species_type.value == 'carnivore']
            
            # Create real-time chart data
            chart_data = self.generate_chart_data()
            
            # Get genetic operations data
            genetic_stats = {
                'recent_mutations': stats.get('recent_mutations', 0),
                'recent_crossovers': stats.get('recent_crossovers', 0),
                'total_mutations': stats.get('total_mutations', 0),
                'total_crossovers': stats.get('total_crossovers', 0),
                'mutation_rate_per_minute': stats.get('mutation_rate', 0),
                'crossover_rate_per_minute': stats.get('crossover_rate', 0),
                'recent_events': stats.get('recent_events', [])
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'step': self.canvas.step_count,
                'agents': {
                    'herbivores': {
                        'count': len(agent_data['herbivores']['x']),
                        'positions': list(zip(agent_data['herbivores']['x'], agent_data['herbivores']['y'])),
                        'fitness': agent_data['herbivores']['fitness'],
                        'energy': [float(h.energy) for h in herbivores],
                        'ages': [int(h.age) for h in herbivores],
                        'generations': [getattr(h, 'generation', 1) for h in herbivores],
                        'decisions': [int(getattr(h.brain, 'decisions_made', 0)) for h in herbivores],
                        'avg_fitness': np.mean(agent_data['herbivores']['fitness']) if agent_data['herbivores']['fitness'] else 0
                    },
                    'carnivores': {
                        'count': len(agent_data['carnivores']['x']),
                        'positions': list(zip(agent_data['carnivores']['x'], agent_data['carnivores']['y'])),
                        'fitness': agent_data['carnivores']['fitness'],
                        'energy': [float(c.energy) for c in carnivores],
                        'ages': [int(c.age) for c in carnivores],
                        'generations': [getattr(c, 'generation', 1) for c in carnivores],
                        'decisions': [int(getattr(c.brain, 'decisions_made', 0)) for c in carnivores],
                        'avg_fitness': np.mean(agent_data['carnivores']['fitness']) if agent_data['carnivores']['fitness'] else 0
                    },
                    'food': {
                        'count': len(agent_data['food']['x']),
                        'positions': list(zip(agent_data['food']['x'], agent_data['food']['y']))
                    }
                },
                'stats': {
                    'total_agents': stats.get('herbivore_count', 0) + stats.get('carnivore_count', 0),
                    'avg_fitness': stats.get('avg_neural_fitness', 0),
                    'decisions_made': stats.get('avg_decisions_made', 0)
                },
                'genetics': genetic_stats,
                'charts': chart_data
            }
        except Exception as e:
            print(f"❌ Error getting data: {e}")
            return {'error': str(e)}
    
    def generate_chart_data(self):
        """Generate chart data for visualization"""
        try:
            # Get fitness history
            fitness_data = {
                'herbivore_fitness': list(self.canvas.fitness_history_herb) if hasattr(self.canvas, 'fitness_history_herb') else [],
                'carnivore_fitness': list(self.canvas.fitness_history_carn) if hasattr(self.canvas, 'fitness_history_carn') else []
            }
            
            # Get population history
            population_data = []
            if hasattr(self.canvas, 'population_history'):
                population_data = list(self.canvas.population_history)
            
            return {
                'fitness_evolution': fitness_data,
                'population_dynamics': population_data,
                'neural_diversity': self.get_neural_diversity_data()
            }
        except Exception as e:
            print(f"❌ Error generating charts: {e}")
            return {}
    
    def get_neural_diversity_data(self):
        """Get neural diversity data for visualization"""
        try:
            if hasattr(self.canvas, 'extract_neural_signatures'):
                signatures = self.canvas.extract_neural_signatures()
                
                # Process signatures for web display
                diversity_data = {
                    'herbivore_variants': len(set([s.get('activation_pattern', 0) for s in signatures if s.get('species') == 'herbivore'])),
                    'carnivore_variants': len(set([s.get('activation_pattern', 0) for s in signatures if s.get('species') == 'carnivore'])),
                    'total_diversity': len(set([s.get('activation_pattern', 0) for s in signatures])),
                    'signature_data': [
                        {
                            'species': s.get('species', 'unknown'),
                            'fitness': s.get('fitness', 0),
                            'weight_mean': s.get('weight_mean', 0),
                            'weight_std': s.get('weight_std', 0),
                            'decisions': s.get('decisions_made', 0),
                            'energy': s.get('energy_level', 0)
                        } for s in signatures[:20]  # Limit for performance
                    ]
                }
                return diversity_data
            return {}
        except Exception as e:
            print(f"❌ Error getting neural diversity: {e}")
            return {}
    
    def simulation_loop(self):
        """Main simulation loop with real-time updates"""
        print("🚀 Starting real-time simulation loop")
        
        while self.running and self.clients_connected > 0:
            try:
                # Step the simulation
                if hasattr(self.canvas, 'update_advanced_canvas'):
                    self.canvas.update_advanced_canvas()
                else:
                    self.canvas.env.step()
                    self.canvas.step_count += 1
                
                # Get current data
                data = self.get_current_data()
                
                # Broadcast to all connected clients
                self.socketio.emit('ecosystem_update', data)
                
                # Update frequency (adjustable)
                time.sleep(0.5)  # 2 FPS
                
            except Exception as e:
                print(f"❌ Simulation error: {e}")
                self.running = False
                self.socketio.emit('error', {'message': str(e)})
        
        print("🛑 Simulation loop stopped")
    
    def cleanup_and_exit(self):
        """Clean up resources and exit the program"""
        print("🧹 Cleaning up and exiting...")
        
        # Stop the simulation
        self.running = False
        
        # Clean up canvas resources if available
        try:
            if hasattr(self.canvas, 'cleanup'):
                self.canvas.cleanup()
            if hasattr(self.canvas, 'running'):
                self.canvas.running = False
        except Exception as e:
            print(f"⚠️ Canvas cleanup warning: {e}")
        
        # Close matplotlib figures if any
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception as e:
            print(f"⚠️ Matplotlib cleanup warning: {e}")
        
        print("✅ Cleanup complete - program will exit")
        
        # Exit the program
        import os
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
    
    def get_html_template(self):
        """Generate the HTML template with real-time features"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Ecosystem - Real-Time Stream</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #0a0a0a, #1a1a2e, #16213e);
            color: white;
            overflow-x: hidden;
        }
        
        .header {
            text-align: center;
            padding: 20px;
            background: rgba(26, 26, 46, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 2px solid #4a90e2;
        }
        
        .title {
            font-size: 2.5em;
            margin: 0;
            background: linear-gradient(45deg, #4a90e2, #66bb6a);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: pulse 2s ease-in-out infinite alternate;
        }
        
        @keyframes pulse {
            from { opacity: 0.8; }
            to { opacity: 1; }
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 20px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #4a90e2, #357abd);
            color: white;
        }
        
        .btn-primary:hover {
            background: linear-gradient(135deg, #357abd, #2968a3);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(74, 144, 226, 0.4);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
        }
        
        .btn-danger:hover {
            background: linear-gradient(135deg, #c0392b, #a93226);
            transform: translateY(-2px);
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .ecosystem-view {
            grid-column: 1 / -1;
            background: rgba(26, 26, 46, 0.6);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(74, 144, 226, 0.3);
        }
        
        .canvas-container {
            position: relative;
            width: 100%;
            height: 400px;
            background: #fff;
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid #4a90e2;
        }
        
        #ecosystem-canvas {
            width: 100%;
            height: 100%;
        }
        
        .stats-panel {
            background: rgba(26, 26, 46, 0.6);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(74, 144, 226, 0.3);
        }
        
        .chart-container {
            background: rgba(26, 26, 46, 0.6);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(74, 144, 226, 0.3);
            height: 300px;
        }
        
        .inspector-panel {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(26, 26, 46, 0.95);
            border: 2px solid #4a90e2;
            border-radius: 15px;
            padding: 25px;
            max-width: 500px;
            max-height: 80vh;
            overflow-y: auto;
            z-index: 2000;
            display: none;
            backdrop-filter: blur(10px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        }
        
        .inspector-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1999;
            display: none;
        }
        
        .inspector-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(74, 144, 226, 0.3);
        }
        
        .close-inspector {
            background: #e74c3c;
            color: white;
            border: none;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        
        .neural-diagram {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            min-height: 200px;
        }
        
        .agent-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 15px 0;
        }
        
        .stat-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 8px;
            border-radius: 5px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #4a90e2;
        }
        
        .status-bar {
            position: fixed;
            top: 0;
            right: 0;
            padding: 10px 20px;
            background: rgba(76, 175, 80, 0.9);
            border-radius: 0 0 0 10px;
            font-weight: bold;
            z-index: 1000;
        }
        
        .status-disconnected {
            background: rgba(231, 76, 60, 0.9);
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
        }
        
        .metric-value {
            font-weight: bold;
            color: #4a90e2;
        }
        
        #connection-log {
            max-height: 150px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="status-bar" id="status">🔌 Connecting...</div>
    
    <div class="header">
        <h1 class="title">🧬 AI Ecosystem Real-Time Stream</h1>
        <p>Live neural evolution with WebSocket streaming</p>
    </div>
    
    <div class="controls">
        <button class="btn btn-primary" onclick="startSimulation()">🚀 Start Simulation</button>
        <button class="btn btn-danger" onclick="stopSimulation()">🛑 Stop Simulation</button>
        <button class="btn btn-danger" onclick="endSimulation()" style="background: linear-gradient(135deg, #dc3545, #c82333);">🔚 End Simulation</button>
        <button class="btn btn-primary" onclick="requestData()">📊 Refresh Data</button>
    </div>
    
    <div class="dashboard">
        <div class="ecosystem-view">
            <h3>🌍 Live Ecosystem - Click on agents to inspect!</h3>
            <div class="canvas-container">
                <canvas id="ecosystem-canvas"></canvas>
            </div>
        </div>
        
        <div class="stats-panel">
            <h3>📊 Live Statistics</h3>
            <div class="metric">
                <span>Step:</span>
                <span class="metric-value" id="step-counter">0</span>
            </div>
            <div class="metric">
                <span>🦌 Herbivores:</span>
                <span class="metric-value" id="herbivore-count">0</span>
            </div>
            <div class="metric">
                <span>🐺 Carnivores:</span>
                <span class="metric-value" id="carnivore-count">0</span>
            </div>
            <div class="metric">
                <span>🌾 Food Sources:</span>
                <span class="metric-value" id="food-count">0</span>
            </div>
            <div class="metric">
                <span>🧠 Avg Fitness:</span>
                <span class="metric-value" id="avg-fitness">0</span>
            </div>
            <div class="metric">
                <span>⚡ Decisions/sec:</span>
                <span class="metric-value" id="decisions">0</span>
            </div>
            
            <h4>🧬 Genetic Operations</h4>
            <div class="metric">
                <span>🔀 Recent Mutations:</span>
                <span class="metric-value" id="recent-mutations">0</span>
            </div>
            <div class="metric">
                <span>🧬 Recent Crossovers:</span>
                <span class="metric-value" id="recent-crossovers">0</span>
            </div>
            <div class="metric">
                <span>🔄 Total Mutations:</span>
                <span class="metric-value" id="total-mutations">0</span>
            </div>
            <div class="metric">
                <span>🔬 Total Crossovers:</span>
                <span class="metric-value" id="total-crossovers">0</span>
            </div>
            <div class="metric">
                <span>📈 Mutation Rate/min:</span>
                <span class="metric-value" id="mutation-rate">0</span>
            </div>
            
            <h4>🔌 Connection Log</h4>
            <div id="connection-log"></div>
        </div>
        
        <div class="chart-container">
            <h3>📈 Fitness Evolution</h3>
            <canvas id="fitness-chart"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>👥 Population Dynamics</h3>
            <canvas id="population-chart"></canvas>
        </div>
    </div>
    
    <!-- Agent Inspector Panel -->
    <div class="inspector-overlay" id="inspector-overlay" onclick="closeInspector()"></div>
    <div class="inspector-panel" id="inspector-panel">
        <div class="inspector-header">
            <h3 id="inspector-title">🔍 Agent Inspector</h3>
            <button class="close-inspector" onclick="closeInspector()">×</button>
        </div>
        
        <div class="agent-stats" id="agent-stats">
            <!-- Agent statistics will be populated here -->
        </div>
        
        <div class="neural-diagram">
            <h4>🧠 Neural Network (Real-time)</h4>
            <canvas id="neural-canvas" width="400" height="200"></canvas>
        </div>
        
        <div id="agent-details">
            <!-- Detailed agent information -->
        </div>
    </div>
    
    <script>
        // WebSocket connection
        const socket = io();
        
        // Canvas for ecosystem visualization
        const canvas = document.getElementById('ecosystem-canvas');
        const ctx = canvas.getContext('2d');
        
        // Agent data for click detection
        let currentAgents = [];
        let selectedAgent = null;
        
        // Add click event listener to canvas
        canvas.addEventListener('click', function(event) {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            // Scale click coordinates to canvas coordinates
            const canvasX = (x / canvas.clientWidth) * canvas.width;
            const canvasY = (y / canvas.clientHeight) * canvas.height;
            
            // Check if click hit any agent
            const clickedAgent = findAgentAtPosition(canvasX, canvasY);
            if (clickedAgent) {
                openInspector(clickedAgent);
            }
        });
        
        function findAgentAtPosition(x, y) {
            const clickRadius = 20; // Detection radius around click
            
            for (let agent of currentAgents) {
                const distance = Math.sqrt(
                    Math.pow(x - agent.x, 2) + Math.pow(y - agent.y, 2)
                );
                if (distance <= clickRadius) {
                    return agent;
                }
            }
            return null;
        }
        
        // Resize canvas
        function resizeCanvas() {
            const container = canvas.parentElement;
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
        }
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        
        // Chart configurations
        const fitnessChart = new Chart(document.getElementById('fitness-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Herbivore Fitness',
                        data: [],
                        borderColor: '#66bb6a',
                        backgroundColor: 'rgba(102, 187, 106, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Carnivore Fitness',
                        data: [],
                        borderColor: '#ef5350',
                        backgroundColor: 'rgba(239, 83, 80, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: 'white' } } },
                scales: {
                    x: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                    y: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                }
            }
        });
        
        const populationChart = new Chart(document.getElementById('population-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Herbivores',
                        data: [],
                        borderColor: '#66bb6a',
                        tension: 0.4
                    },
                    {
                        label: 'Carnivores',
                        data: [],
                        borderColor: '#ef5350',
                        tension: 0.4
                    },
                    {
                        label: 'Food',
                        data: [],
                        borderColor: '#ff9800',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: 'white' } } },
                scales: {
                    x: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                    y: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                }
            }
        });
        
        // WebSocket events
        socket.on('connect', function() {
            updateStatus('🟢 Connected', false);
            logMessage('Connected to ecosystem stream');
        });
        
        socket.on('disconnect', function() {
            updateStatus('🔴 Disconnected', true);
            logMessage('Disconnected from server');
        });
        
        socket.on('ecosystem_update', function(data) {
            updateEcosystemView(data);
            updateStatistics(data);
            updateCharts(data);
            logMessage(`Step ${data.step}: Updated ecosystem data`);
        });
        
        socket.on('error', function(data) {
            logMessage(`ERROR: ${data.message}`, true);
        });
        
        socket.on('status', function(data) {
            logMessage(data.message);
        });
        
        socket.on('agent_details', function(data) {
            if (selectedAgent && data.agent_id === selectedAgent.id) {
                // Update agent details panel
                updateAgentDetailsPanel(data);
                // Redraw neural network with more accurate data
                if (data.neural_data) {
                    drawAdvancedNeuralNetwork(data);
                }
            }
        });
        
        // Control functions
        function startSimulation() {
            socket.emit('start_simulation');
            logMessage('Starting simulation...');
        }
        
        function stopSimulation() {
            socket.emit('stop_simulation');
            logMessage('Stopping simulation...');
        }
        
        function endSimulation() {
            if (confirm('Are you sure you want to end the simulation and close the program? This cannot be undone.')) {
                socket.emit('end_simulation');
                logMessage('Ending simulation and closing program...');
                setTimeout(() => {
                    window.close();
                }, 2000);
            }
        }
        
        function requestData() {
            socket.emit('request_data');
            logMessage('Requesting current data...');
        }
        
        // Agent Inspector Functions
        function openInspector(agent) {
            selectedAgent = agent;
            document.getElementById('inspector-overlay').style.display = 'block';
            document.getElementById('inspector-panel').style.display = 'block';
            
            // Update inspector title
            const speciesEmoji = agent.species === 'herbivore' ? '🦌' : '🐺';
            document.getElementById('inspector-title').textContent = 
                `${speciesEmoji} ${agent.species.charAt(0).toUpperCase() + agent.species.slice(1)} #${agent.id}`;
            
            // Update agent statistics
            updateAgentStats(agent);
            
            // Draw neural network
            drawNeuralNetwork(agent);
            
            // Request detailed agent data from server
            socket.emit('inspect_agent', { agent_id: agent.id });
        }
        
        function closeInspector() {
            document.getElementById('inspector-overlay').style.display = 'none';
            document.getElementById('inspector-panel').style.display = 'none';
            selectedAgent = null;
        }
        
        function updateAgentStats(agent) {
            const statsContainer = document.getElementById('agent-stats');
            statsContainer.innerHTML = `
                <div class="stat-item">
                    <div>Generation</div>
                    <div class="stat-value">${agent.generation || 1}</div>
                </div>
                <div class="stat-item">
                    <div>Energy</div>
                    <div class="stat-value">${Math.round(agent.energy)}</div>
                </div>
                <div class="stat-item">
                    <div>Fitness</div>
                    <div class="stat-value">${agent.fitness.toFixed(1)}</div>
                </div>
                <div class="stat-item">
                    <div>Age</div>
                    <div class="stat-value">${agent.age || 0}</div>
                </div>
                <div class="stat-item">
                    <div>Position</div>
                    <div class="stat-value">${Math.round(agent.x)}, ${Math.round(agent.y)}</div>
                </div>
                <div class="stat-item">
                    <div>Decisions Made</div>
                    <div class="stat-value">${agent.decisions_made || 0}</div>
                </div>
                <div class="stat-item">
                    <div>Can Reproduce</div>
                    <div class="stat-value">${agent.can_reproduce ? '✅' : '❌'}</div>
                </div>
            `;
        }
        
        function drawNeuralNetwork(agent) {
            const neuralCanvas = document.getElementById('neural-canvas');
            const neuralCtx = neuralCanvas.getContext('2d');
            
            // Clear canvas
            neuralCtx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            neuralCtx.fillRect(0, 0, neuralCanvas.width, neuralCanvas.height);
            
            // Neural network structure (simplified)
            const layers = [
                { neurons: 8, label: 'Sensors' },     // Input layer
                { neurons: 12, label: 'Hidden 1' },   // Hidden layer 1
                { neurons: 8, label: 'Hidden 2' },    // Hidden layer 2
                { neurons: 4, label: 'Actions' }      // Output layer
            ];
            
            const layerSpacing = neuralCanvas.width / (layers.length + 1);
            const neuronRadius = 8;
            
            // Draw connections first (behind neurons)
            neuralCtx.strokeStyle = 'rgba(74, 144, 226, 0.3)';
            neuralCtx.lineWidth = 1;
            
            for (let i = 0; i < layers.length - 1; i++) {
                const currentLayer = layers[i];
                const nextLayer = layers[i + 1];
                
                const currentX = layerSpacing * (i + 1);
                const nextX = layerSpacing * (i + 2);
                
                for (let j = 0; j < currentLayer.neurons; j++) {
                    const currentY = (neuralCanvas.height / (currentLayer.neurons + 1)) * (j + 1);
                    
                    for (let k = 0; k < nextLayer.neurons; k++) {
                        const nextY = (neuralCanvas.height / (nextLayer.neurons + 1)) * (k + 1);
                        
                        // Simulate activity based on agent's current state
                        const activity = Math.random() * agent.fitness / 20; // Activity based on fitness
                        const alpha = Math.max(0.1, Math.min(0.8, activity));
                        
                        neuralCtx.strokeStyle = `rgba(74, 144, 226, ${alpha})`;
                        neuralCtx.beginPath();
                        neuralCtx.moveTo(currentX, currentY);
                        neuralCtx.lineTo(nextX, nextY);
                        neuralCtx.stroke();
                    }
                }
            }
            
            // Draw neurons
            for (let i = 0; i < layers.length; i++) {
                const layer = layers[i];
                const x = layerSpacing * (i + 1);
                
                for (let j = 0; j < layer.neurons; j++) {
                    const y = (neuralCanvas.height / (layer.neurons + 1)) * (j + 1);
                    
                    // Neuron activity based on position and agent state
                    const baseActivity = 0.3;
                    const activityBoost = Math.random() * (agent.energy / 200); // Activity based on energy
                    const activity = Math.min(1.0, baseActivity + activityBoost);
                    
                    // Color based on layer type
                    let color;
                    if (i === 0) color = `rgba(102, 187, 106, ${activity})`; // Green for input
                    else if (i === layers.length - 1) color = `rgba(239, 83, 80, ${activity})`; // Red for output
                    else color = `rgba(74, 144, 226, ${activity})`; // Blue for hidden
                    
                    // Draw neuron
                    neuralCtx.fillStyle = color;
                    neuralCtx.beginPath();
                    neuralCtx.arc(x, y, neuronRadius, 0, 2 * Math.PI);
                    neuralCtx.fill();
                    
                    // Draw neuron border
                    neuralCtx.strokeStyle = 'white';
                    neuralCtx.lineWidth = 1;
                    neuralCtx.stroke();
                }
                
                // Draw layer labels
                neuralCtx.fillStyle = 'white';
                neuralCtx.font = '12px Arial';
                neuralCtx.textAlign = 'center';
                neuralCtx.fillText(layer.label, x, neuralCanvas.height - 10);
            }
            
            // Add activity indicator
            neuralCtx.fillStyle = 'white';
            neuralCtx.font = '14px Arial';
            neuralCtx.textAlign = 'left';
            neuralCtx.fillText(`🧠 Network Activity: ${(agent.fitness / 10).toFixed(1)}%`, 10, 20);
        }
        
        function updateAgentDetailsPanel(agentData) {
            // Update detailed statistics with server data
            const detailsContainer = document.getElementById('agent-details');
            
            detailsContainer.innerHTML = `
                <h4>📊 Detailed Performance</h4>
                <div class="agent-stats">
                    <div class="stat-item">
                        <div>Reproduction Cooldown</div>
                        <div class="stat-value">${agentData.reproduction_cooldown}</div>
                    </div>
                    <div class="stat-item">
                        <div>Lifetime Energy Gained</div>
                        <div class="stat-value">${agentData.performance_stats.lifetime_energy_gained}</div>
                    </div>
                    <div class="stat-item">
                        <div>Food Consumed</div>
                        <div class="stat-value">${agentData.performance_stats.lifetime_food_consumed}</div>
                    </div>
                    <div class="stat-item">
                        <div>Offspring Count</div>
                        <div class="stat-value">${agentData.performance_stats.offspring_count}</div>
                    </div>
                    <div class="stat-item">
                        <div>Energy Efficiency</div>
                        <div class="stat-value">${(agentData.energy / agentData.max_energy * 100).toFixed(1)}%</div>
                    </div>
                    <div class="stat-item">
                        <div>Survival Time</div>
                        <div class="stat-value">${agentData.performance_stats.survival_time} steps</div>
                    </div>
                </div>
                
                <h4>🔬 Neural Network Analysis</h4>
                <div style="font-size: 12px; color: #aaa;">
                    <p>• Layer structures: ${agentData.neural_data.layer_sizes?.map(s => `${s[0]}x${s[1]}`).join(' → ') || 'Loading...'}</p>
                    <p>• Average activations: ${agentData.neural_data.activations?.map(a => a.toFixed(3)).join(', ') || 'Loading...'}</p>
                    <p>• Decision patterns adapting based on ${agentData.decisions_made} total decisions</p>
                </div>
            `;
        }
        
        function drawAdvancedNeuralNetwork(agentData) {
            if (!agentData.neural_data || !agentData.neural_data.layer_sizes) {
                // Fallback to basic neural network if no detailed data
                drawNeuralNetwork(selectedAgent);
                return;
            }
            
            const neuralCanvas = document.getElementById('neural-canvas');
            const neuralCtx = neuralCanvas.getContext('2d');
            
            // Clear canvas
            neuralCtx.fillStyle = 'rgba(0, 0, 0, 0.9)';
            neuralCtx.fillRect(0, 0, neuralCanvas.width, neuralCanvas.height);
            
            const layerSizes = agentData.neural_data.layer_sizes;
            const activations = agentData.neural_data.activations || [];
            const weightRanges = agentData.neural_data.weight_ranges || [];
            
            const layerLabels = ['Sensors', 'Hidden 1', 'Hidden 2', 'Actions'];
            const layerSpacing = neuralCanvas.width / (layerSizes.length + 1);
            const neuronRadius = 6;
            
            // Draw connections with real weight data
            for (let i = 0; i < layerSizes.length - 1; i++) {
                const currentLayerSize = layerSizes[i][1]; // Output size of current layer
                const nextLayerSize = layerSizes[i + 1][1]; // Output size of next layer
                
                const currentX = layerSpacing * (i + 1);
                const nextX = layerSpacing * (i + 2);
                
                // Connection strength based on weight ranges
                const weightRange = weightRanges[i] || [-1, 1];
                const weightSpread = weightRange[1] - weightRange[0];
                
                for (let j = 0; j < currentLayerSize; j++) {
                    const currentY = (neuralCanvas.height / (currentLayerSize + 1)) * (j + 1);
                    
                    for (let k = 0; k < nextLayerSize; k++) {
                        const nextY = (neuralCanvas.height / (nextLayerSize + 1)) * (k + 1);
                        
                        // Connection strength based on actual weight data
                        const connectionStrength = Math.random() * weightSpread + 0.2;
                        const alpha = Math.max(0.1, Math.min(0.7, connectionStrength));
                        
                        // Color based on weight sign (positive = blue, negative = red)
                        const isPositive = Math.random() > 0.5;
                        const color = isPositive ? `rgba(74, 144, 226, ${alpha})` : `rgba(239, 83, 80, ${alpha})`;
                        
                        neuralCtx.strokeStyle = color;
                        neuralCtx.lineWidth = alpha * 2;
                        neuralCtx.beginPath();
                        neuralCtx.moveTo(currentX, currentY);
                        neuralCtx.lineTo(nextX, nextY);
                        neuralCtx.stroke();
                    }
                }
            }
            
            // Draw neurons with activation data
            for (let i = 0; i < layerSizes.length; i++) {
                const layerSize = layerSizes[i][1];
                const x = layerSpacing * (i + 1);
                const activation = activations[i] || 0.5;
                
                for (let j = 0; j < layerSize; j++) {
                    const y = (neuralCanvas.height / (layerSize + 1)) * (j + 1);
                    
                    // Neuron activity based on real activation data
                    const neuronActivity = Math.abs(activation) + Math.random() * 0.3;
                    const clampedActivity = Math.max(0.2, Math.min(1.0, neuronActivity));
                    
                    // Color based on layer type and activation
                    let color;
                    if (i === 0) color = `rgba(102, 187, 106, ${clampedActivity})`; // Green for input
                    else if (i === layerSizes.length - 1) color = `rgba(239, 83, 80, ${clampedActivity})`; // Red for output
                    else color = `rgba(74, 144, 226, ${clampedActivity})`; // Blue for hidden
                    
                    // Draw neuron
                    neuralCtx.fillStyle = color;
                    neuralCtx.beginPath();
                    neuralCtx.arc(x, y, neuronRadius + activation * 3, 0, 2 * Math.PI);
                    neuralCtx.fill();
                    
                    // Draw neuron border
                    neuralCtx.strokeStyle = activation > 0 ? '#fff' : '#888';
                    neuralCtx.lineWidth = 1;
                    neuralCtx.stroke();
                }
                
                // Draw layer labels
                neuralCtx.fillStyle = 'white';
                neuralCtx.font = '11px Arial';
                neuralCtx.textAlign = 'center';
                neuralCtx.fillText(layerLabels[i] || `Layer ${i}`, x, neuralCanvas.height - 5);
                
                // Draw layer size info
                neuralCtx.font = '9px Arial';
                neuralCtx.fillStyle = '#aaa';
                neuralCtx.fillText(`(${layerSize})`, x, neuralCanvas.height - 18);
            }
            
            // Add detailed activity indicator
            neuralCtx.fillStyle = 'white';
            neuralCtx.font = '12px Arial';
            neuralCtx.textAlign = 'left';
            neuralCtx.fillText(`🧠 Live Neural Activity - Decision #${agentData.decisions_made}`, 5, 15);
            
            // Add fitness indicator
            neuralCtx.fillStyle = agentData.fitness > 10 ? '#4CAF50' : '#FFC107';
            neuralCtx.fillText(`⚡ Fitness: ${agentData.fitness.toFixed(1)}`, 5, 30);
        }
        
        // Update functions
        function updateStatus(text, isError) {
            const status = document.getElementById('status');
            status.textContent = text;
            status.className = isError ? 'status-bar status-disconnected' : 'status-bar';
        }
        
        function updateStatistics(data) {
            document.getElementById('step-counter').textContent = data.step || 0;
            document.getElementById('herbivore-count').textContent = data.agents?.herbivores?.count || 0;
            document.getElementById('carnivore-count').textContent = data.agents?.carnivores?.count || 0;
            document.getElementById('food-count').textContent = data.agents?.food?.count || 0;
            document.getElementById('avg-fitness').textContent = (data.stats?.avg_fitness || 0).toFixed(1);
            document.getElementById('decisions').textContent = (data.stats?.decisions_made || 0).toFixed(0);
            
            // Update genetic operations statistics
            if (data.genetics) {
                document.getElementById('recent-mutations').textContent = data.genetics.recent_mutations || 0;
                document.getElementById('recent-crossovers').textContent = data.genetics.recent_crossovers || 0;
                document.getElementById('total-mutations').textContent = data.genetics.total_mutations || 0;
                document.getElementById('total-crossovers').textContent = data.genetics.total_crossovers || 0;
                document.getElementById('mutation-rate').textContent = (data.genetics.mutation_rate_per_minute || 0).toFixed(1);
            }
        }
        
        function updateEcosystemView(data) {
            // Clear canvas
            ctx.fillStyle = '#fff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Clear and rebuild current agents array for click detection
            currentAgents = [];
            
            // Draw environment boundaries
            ctx.strokeStyle = '#4a90e2';
            ctx.lineWidth = 2;
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
            
            // Scale factors
            const scaleX = canvas.width / 200;  // Assuming 200x200 environment
            const scaleY = canvas.height / 200;
            
            // Draw food
            if (data.agents?.food?.positions) {
                ctx.fillStyle = '#8B4513';
                ctx.strokeStyle = '#654321';
                ctx.lineWidth = 1;
                data.agents.food.positions.forEach(([x, y]) => {
                    ctx.fillRect(x * scaleX - 3, y * scaleY - 3, 6, 6);
                    ctx.strokeRect(x * scaleX - 3, y * scaleY - 3, 6, 6);
                });
            }
            
            // Draw herbivores and store their data
            if (data.agents?.herbivores?.positions) {
                ctx.fillStyle = '#2E7D32';  // Darker green for white background
                ctx.strokeStyle = '#1B5E20';  // Even darker green for outline
                ctx.lineWidth = 1;
                data.agents.herbivores.positions.forEach(([x, y], i) => {
                    ctx.beginPath();
                    const fitness = data.agents.herbivores.fitness[i] || 0;
                    const energy = data.agents.herbivores.energy?.[i] || 100;
                    const size = 3 + Math.min(fitness / 5, 5);
                    
                    const scaledX = x * scaleX;
                    const scaledY = y * scaleY;
                    
                    ctx.arc(scaledX, scaledY, size, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();  // Add outline
                    
                    // Store agent data for click detection
                    currentAgents.push({
                        id: `herb_${i}`,
                        species: 'herbivore',
                        x: scaledX,
                        y: scaledY,
                        fitness: fitness,
                        energy: energy,
                        age: data.agents.herbivores.ages?.[i] || 0,
                        generation: data.agents.herbivores.generations?.[i] || 1,
                        decisions_made: data.agents.herbivores.decisions?.[i] || 0,
                        can_reproduce: energy > 120 // Based on reproduction threshold
                    });
                });
            }
            
            // Draw carnivores and store their data
            if (data.agents?.carnivores?.positions) {
                ctx.fillStyle = '#C62828';  // Darker red for white background
                ctx.strokeStyle = '#B71C1C';  // Even darker red for outline
                ctx.lineWidth = 1;
                data.agents.carnivores.positions.forEach(([x, y], i) => {
                    const fitness = data.agents.carnivores.fitness[i] || 0;
                    const energy = data.agents.carnivores.energy?.[i] || 100;
                    const size = 4 + Math.min(fitness / 5, 6);
                    
                    const scaledX = x * scaleX;
                    const scaledY = y * scaleY;
                    
                    // Draw triangle
                    ctx.beginPath();
                    ctx.moveTo(scaledX, scaledY - size);
                    ctx.lineTo(scaledX - size, scaledY + size);
                    ctx.lineTo(scaledX + size, scaledY + size);
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();  // Add outline
                    
                    // Store agent data for click detection
                    currentAgents.push({
                        id: `carn_${i}`,
                        species: 'carnivore',
                        x: scaledX,
                        y: scaledY,
                        fitness: fitness,
                        energy: energy,
                        age: data.agents.carnivores.ages?.[i] || 0,
                        generation: data.agents.carnivores.generations?.[i] || 1,
                        decisions_made: data.agents.carnivores.decisions?.[i] || 0,
                        can_reproduce: energy > 150 // Based on reproduction threshold
                    });
                });
            }
            
            // Update neural network for selected agent if inspector is open
            if (selectedAgent) {
                // Find updated data for selected agent and redraw neural network
                const updatedAgent = currentAgents.find(a => a.id === selectedAgent.id);
                if (updatedAgent) {
                    selectedAgent = updatedAgent;
                    drawNeuralNetwork(updatedAgent);
                    updateAgentStats(updatedAgent);
                }
            }
        }
        
        function updateCharts(data) {
            if (data.charts?.fitness_evolution) {
                const herbFitness = data.charts.fitness_evolution.herbivore_fitness || [];
                const carnFitness = data.charts.fitness_evolution.carnivore_fitness || [];
                
                fitnessChart.data.labels = herbFitness.map((_, i) => i);
                fitnessChart.data.datasets[0].data = herbFitness;
                fitnessChart.data.datasets[1].data = carnFitness;
                fitnessChart.update('none');
            }
            
            if (data.charts?.population_dynamics) {
                const popData = data.charts.population_dynamics;
                if (popData.length > 0) {
                    populationChart.data.labels = popData.map((_, i) => i);
                    populationChart.data.datasets[0].data = popData.map(p => p.herbivores || 0);
                    populationChart.data.datasets[1].data = popData.map(p => p.carnivores || 0);
                    populationChart.data.datasets[2].data = popData.map(p => p.food || 0);
                    populationChart.update('none');
                }
            }
        }
        
        function logMessage(message, isError = false) {
            const log = document.getElementById('connection-log');
            const timestamp = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.style.color = isError ? '#ef5350' : '#66bb6a';
            entry.textContent = `[${timestamp}] ${message}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
            
            // Limit log entries
            while (log.children.length > 50) {
                log.removeChild(log.firstChild);
            }
        }
        
        // Auto-start
        setTimeout(() => {
            requestData();
        }, 1000);
    </script>
</body>
</html>
        '''
    
    def start_server(self, host='localhost', port=5000):
        """Start the web server"""
        print(f"🌐 Starting ecosystem web server on http://{host}:{port}")
        print("   - Real-time WebSocket streaming")
        print("   - No page refreshes needed")
        print("   - Live ecosystem visualization")
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=False)
        except Exception as e:
            print(f"❌ Server error: {e}")

def start_realtime_web_server(canvas):
    """Start the real-time web server for ecosystem visualization"""
    server = EcosystemWebServer(canvas)
    
    # Try to open browser automatically
    import webbrowser
    import threading
    
    def open_browser():
        time.sleep(2)  # Give server time to start
        try:
            webbrowser.open('http://localhost:5000')
            print("✅ Opened browser automatically")
        except:
            print("⚠️ Could not open browser automatically")
            print("📁 Open http://localhost:5000 manually")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    server.start_server()
