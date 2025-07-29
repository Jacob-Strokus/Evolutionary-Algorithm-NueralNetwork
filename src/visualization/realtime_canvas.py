"""
Real-Time 2D Ecosystem Canvas
Interactive visualization showing agents, food, and neural network activity in real-time
"""
import matplotlib
import os
import subprocess

# Try multiple approaches to get a working GUI display
def setup_display():
    """Setup display for WSL/Linux environments"""
    
    # Method 1: Check if we're in WSL and try to use Windows X server
    if os.environ.get('WSL_DISTRO_NAME'):
        print("🔍 Detected WSL environment")
        
        # Try to use Windows X server (VcXsrv, Xming, etc.)
        try:
            # Check if DISPLAY is set
            if 'DISPLAY' not in os.environ:
                # Try common Windows X server configurations
                os.environ['DISPLAY'] = ':0'
                print("🔧 Set DISPLAY to :0 for Windows X server")
            
            # Test if X server is accessible
            result = subprocess.run(['xset', 'q'], capture_output=True, timeout=2)
            if result.returncode == 0:
                print("✅ X server detected - using GUI mode")
                matplotlib.use('TkAgg')
                return False  # Not headless
            else:
                print("⚠️ X server not accessible")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️ X utilities not found")
    
    # Method 2: Try Qt backend (often works better in WSL)
    try:
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        
        # Test if we can create a figure
        fig = plt.figure(figsize=(1, 1))
        plt.close(fig)
        print("✅ Qt5Agg backend working - using GUI mode")
        return False  # Not headless
    except:
        pass
    
    # Method 3: Try TkAgg backend
    try:
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        
        # Test if we can create a figure
        fig = plt.figure(figsize=(1, 1))
        plt.close(fig)
        print("✅ TkAgg backend working - using GUI mode")
        return False  # Not headless
    except:
        pass
    
    # Method 4: Fallback to web-based display using webbrowser
    try:
        import webbrowser
        matplotlib.use('Agg')
        print("🌐 Falling back to web-based display mode")
        return "web"  # Web mode
    except:
        pass
    
    # Final fallback: headless mode
    print("� No GUI available - using headless mode")
    matplotlib.use('Agg')
    return True  # Headless

# Setup display mode
DISPLAY_MODE = setup_display()
HEADLESS_MODE = (DISPLAY_MODE is True)
WEB_MODE = (DISPLAY_MODE == "web")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType
import time
import threading
from collections import deque

class RealTimeEcosystemCanvas:
    """Real-time 2D visualization of the neural ecosystem"""
    
    def __init__(self, width=200, height=200, update_interval=100):
        self.width = width
        self.height = height
        self.update_interval = update_interval  # milliseconds
        
        # Create environment
        self.env = NeuralEnvironment(width=width, height=height, use_neural_agents=True)
        
        # Set up matplotlib for real-time plotting
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('lightblue')  # Sky blue background
        
        # Initialize plot elements
        self.herbivore_scatter = None
        self.carnivore_scatter = None
        self.food_scatter = None
        self.fitness_text = None
        self.stats_text = None
        
        # Performance tracking
        self.fitness_history = deque(maxlen=100)
        self.step_count = 0
        
        # Animation control
        self.running = True
        self.paused = False
        
        self.setup_canvas()
    
    def setup_canvas(self):
        """Set up the initial canvas layout"""
        self.ax.set_title('AI Neural Ecosystem Canvas', fontsize=16, fontweight='bold', pad=20)
        self.ax.set_xlabel('Environment Width', fontsize=12)
        self.ax.set_ylabel('Environment Height', fontsize=12)
        
        # Add grid for better spatial reference
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        # Create legend elements
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Herbivores'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Carnivores'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='brown', markersize=8, label='Food')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Initialize text displays
        self.stats_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                      verticalalignment='top', fontsize=10, 
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.fitness_text = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes, 
                                       verticalalignment='bottom', fontsize=10,
                                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def get_agent_data(self):
        """Extract current agent positions and fitness data"""
        neural_agents = [agent for agent in self.env.agents if isinstance(agent, NeuralAgent)]
        
        herbivores = [a for a in neural_agents if a.species_type == SpeciesType.HERBIVORE]
        carnivores = [a for a in neural_agents if a.species_type == SpeciesType.CARNIVORE]
        
        # Herbivore data
        herb_x = [a.position.x for a in herbivores]
        herb_y = [a.position.y for a in herbivores]
        herb_fitness = [a.brain.fitness_score for a in herbivores]
        herb_energy = [a.energy / a.max_energy for a in herbivores]  # Normalized energy
        
        # Carnivore data
        carn_x = [a.position.x for a in carnivores]
        carn_y = [a.position.y for a in carnivores]
        carn_fitness = [a.brain.fitness_score for a in carnivores]
        carn_energy = [a.energy / a.max_energy for a in carnivores]  # Normalized energy
        
        # Food data
        food_x = [f.position.x for f in self.env.food_sources if f.is_available]
        food_y = [f.position.y for f in self.env.food_sources if f.is_available]
        
        return {
            'herbivores': {'x': herb_x, 'y': herb_y, 'fitness': herb_fitness, 'energy': herb_energy},
            'carnivores': {'x': carn_x, 'y': carn_y, 'fitness': carn_fitness, 'energy': carn_energy},
            'food': {'x': food_x, 'y': food_y}
        }
    
    def update_canvas(self):
        """Update the canvas with current ecosystem state"""
        # Step the simulation
        self.env.step()
        self.step_count += 1
        
        # Get current data
        data = self.get_agent_data()
        
        # Clear previous scatter plots
        if self.herbivore_scatter:
            self.herbivore_scatter.remove()
        if self.carnivore_scatter:
            self.carnivore_scatter.remove()
        if self.food_scatter:
            self.food_scatter.remove()
        
        # Plot food sources
        if data['food']['x']:
            self.food_scatter = self.ax.scatter(data['food']['x'], data['food']['y'], 
                                              c='brown', s=60, marker='s', alpha=0.7, 
                                              label='Food', edgecolors='darkbrown', linewidth=1)
        
        # Plot herbivores with fitness-based coloring
        if data['herbivores']['x']:
            # Size based on energy, color based on fitness
            herb_sizes = [50 + 30 * energy for energy in data['herbivores']['energy']]
            self.herbivore_scatter = self.ax.scatter(data['herbivores']['x'], data['herbivores']['y'],
                                                   c=data['herbivores']['fitness'], s=herb_sizes,
                                                   cmap='Greens', marker='o', alpha=0.8,
                                                   vmin=0, vmax=max(50, max(data['herbivores']['fitness']) if data['herbivores']['fitness'] else 50),
                                                   edgecolors='darkgreen', linewidth=1)
        
        # Plot carnivores with fitness-based coloring
        if data['carnivores']['x']:
            # Size based on energy, color based on fitness
            carn_sizes = [60 + 40 * energy for energy in data['carnivores']['energy']]
            self.carnivore_scatter = self.ax.scatter(data['carnivores']['x'], data['carnivores']['y'],
                                                   c=data['carnivores']['fitness'], s=carn_sizes,
                                                   cmap='Reds', marker='^', alpha=0.8,
                                                   vmin=0, vmax=max(50, max(data['carnivores']['fitness']) if data['carnivores']['fitness'] else 50),
                                                   edgecolors='darkred', linewidth=1)
        
        # Update statistics
        stats = self.env.get_neural_stats()
        avg_fitness = stats.get('avg_neural_fitness', 0)
        self.fitness_history.append(avg_fitness)
        
        # Update text displays
        stats_text = f"""Ecosystem Status (Step {self.step_count})
Herbivores: {len(data['herbivores']['x'])}
Carnivores: {len(data['carnivores']['x'])}
Food Available: {len(data['food']['x'])}
Avg Fitness: {avg_fitness:.1f}
Decisions/Agent: {stats.get('avg_decisions_made', 0):.0f}"""
        
        self.stats_text.set_text(stats_text)
        
        # Fitness trend analysis
        if len(self.fitness_history) > 10:
            recent_trend = np.mean(list(self.fitness_history)[-5:]) - np.mean(list(self.fitness_history)[:5])
            trend_text = f"Learning Trend: {recent_trend:+.1f}"
            if recent_trend > 1:
                trend_text += " - Learning!"
            elif recent_trend > 0:
                trend_text += " - Improving"
            else:
                trend_text += " - Stable"
        else:
            trend_text = "Collecting data..."
        
        # Add genetic operations info
        mutations_recent = stats.get('recent_mutations', 0)
        crossovers_recent = stats.get('recent_crossovers', 0)
        mutations_total = stats.get('total_mutations', 0)
        crossovers_total = stats.get('total_crossovers', 0)
        
        fitness_text = f"""Neural Network Activity
{trend_text}
Best Herbivore: {max(data['herbivores']['fitness']) if data['herbivores']['fitness'] else 0:.1f}
Best Carnivore: {max(data['carnivores']['fitness']) if data['carnivores']['fitness'] else 0:.1f}

🧬 Genetic Operations:
Mutations: {mutations_recent} recent, {mutations_total} total
Crossovers: {crossovers_recent} recent, {crossovers_total} total
Rate: {stats.get('mutation_rate', 0)}/min mutations
Agent Sizes: Energy Level
Colors: Fitness Score"""
        
        self.fitness_text.set_text(fitness_text)
        
        # Reset counters for next cycle
        self.env.genetic_algorithm.reset_recent_counters()
        
        return True
    
    def animate(self, frame):
        """Animation function for matplotlib"""
        if not self.paused and self.running:
            return self.update_canvas()
    
    def start_animation(self, save_animation=False):
        """Start the real-time animation"""
        print("Starting Real-Time Ecosystem Canvas")
        print("=" * 50)
        
        if WEB_MODE:
            print("🌐 Starting modern real-time web display")
            self.start_modern_web_display()
            return
        elif HEADLESS_MODE:
            print("📱 Running in headless mode - creating static snapshots")
            print("   - Will run simulation and save images")
            print("   - Progress shown in terminal")
            
            # Run simulation and save snapshots
            for step in range(100):  # Run 100 steps
                self.update_canvas()
                
                # Save snapshot every 10 steps
                if step % 10 == 0:
                    filename = f'ecosystem_step_{step:03d}.png'
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"   Saved: {filename}")
            
            print("Simulation complete! Check saved PNG files.")
            return
        
        print("🖥️ Starting GUI window display")
        print("Controls:")
        print("   - Close window to stop")
        print("   - Agents move and learn in real-time")
        print("   - Colors show fitness, sizes show energy")
        print("=" * 50)
        
        # For GUI mode, enable interactive plotting
        plt.ion()
        
        # Create animation with cache_frame_data=False to avoid warnings
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=self.update_interval, 
                                         blit=False, repeat=True, cache_frame_data=False)
        
        # Set up close event
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        if save_animation:
            print("💾 Saving animation (this may take a while)...")
            self.ani.save('ecosystem_animation.gif', writer='pillow', fps=10)
            print("✅ Animation saved as 'ecosystem_animation.gif'")
        
        # Show the window
        plt.show()
        
        # Keep the plot alive and responsive
        try:
            while self.running:
                plt.pause(0.1)
                # Force window update
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        except KeyboardInterrupt:
            print("\n🛑 Animation stopped by user")
        
        self.cleanup()
    
    def start_web_animation(self):
        """Start web-based animation using HTML output"""
        import tempfile
        import webbrowser
        import time
        
        print("🌐 Creating smooth web-based real-time display...")
        
        # Create enhanced HTML file with smooth JavaScript polling
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Ecosystem Real-Time Display</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    text-align: center; 
                    background: linear-gradient(135deg, #1e1e1e, #2d2d2d); 
                    color: white; 
                    margin: 0; 
                    padding: 20px;
                    overflow-x: hidden;
                }
                .container { max-width: 1200px; margin: 0 auto; }
                img { 
                    max-width: 95%; 
                    height: auto; 
                    border: 3px solid #444; 
                    border-radius: 15px; 
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                    transition: opacity 0.5s ease-in-out;
                    background: #333;
                }
                .img-loading { opacity: 0.7; }
                .status { 
                    margin: 20px; 
                    padding: 15px; 
                    background: rgba(51, 51, 51, 0.8); 
                    border-radius: 10px; 
                    backdrop-filter: blur(10px);
                    border: 1px solid #555;
                }
                .title { 
                    color: #4CAF50; 
                    font-size: 28px; 
                    margin-bottom: 10px; 
                    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
                    animation: pulse 2s ease-in-out infinite alternate;
                }
                @keyframes pulse {
                    from { text-shadow: 0 2px 4px rgba(76, 175, 80, 0.3); }
                    to { text-shadow: 0 2px 8px rgba(76, 175, 80, 0.6); }
                }
                .controls {
                    margin: 20px;
                    display: flex;
                    justify-content: center;
                    gap: 15px;
                    flex-wrap: wrap;
                }
                .control-btn {
                    padding: 8px 16px;
                    background: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                    transition: all 0.3s ease;
                }
                .control-btn:hover { background: #45a049; transform: translateY(-2px); }
                .control-btn.active { background: #FF9800; }
                .progress-bar {
                    width: 100%;
                    height: 4px;
                    background: #333;
                    border-radius: 2px;
                    overflow: hidden;
                    margin: 10px 0;
                }
                .progress-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #4CAF50, #8BC34A);
                    width: 0%;
                    transition: width 0.3s ease;
                }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }
                .stat-card {
                    background: rgba(51, 51, 51, 0.6);
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px solid #555;
                }
                .connection-status {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 8px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: bold;
                }
                .connected { background: #4CAF50; }
                .disconnected { background: #f44336; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="connection-status connected" id="status">🟢 Connected</div>
                
                <div class="title">🧬 AI Ecosystem Real-Time Display</div>
                
                <div class="controls">
                    <button class="control-btn active" onclick="setRefreshRate(1000)" id="fast">Fast (1s)</button>
                    <button class="control-btn" onclick="setRefreshRate(2000)" id="normal">Normal (2s)</button>
                    <button class="control-btn" onclick="setRefreshRate(3000)" id="slow">Slow (3s)</button>
                    <button class="control-btn" onclick="togglePause()" id="pauseBtn">Pause</button>
                </div>
                
                <div class="status">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressBar"></div>
                    </div>
                    <p>Step: <span id="step">0</span> | Next update in: <span id="countdown">1</span>s</p>
                </div>
                
                <img id="ecosystem" src="realtime_ecosystem.png" alt="Ecosystem Display" onload="imageLoaded()">
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>🦌 Herbivores</h3>
                        <p>Plant-eating agents</p>
                        <p>Green circles, size = energy</p>
                    </div>
                    <div class="stat-card">
                        <h3>🐺 Carnivores</h3>
                        <p>Predator agents</p>
                        <p>Red triangles, size = energy</p>
                    </div>
                    <div class="stat-card">
                        <h3>🌾 Food Sources</h3>
                        <p>Energy resources</p>
                        <p>Brown squares</p>
                    </div>
                    <div class="stat-card">
                        <h3>🧠 Neural Learning</h3>
                        <p>AI decision making</p>
                        <p>Colors show fitness scores</p>
                    </div>
                </div>
                
                <div class="status">
                    <p><strong>Real-time Neural Learning in Progress</strong></p>
                    <p>Watch agents become smarter through experience!</p>
                </div>
            </div>
            
            <script>
                let refreshRate = 1000; // Default 1 second
                let isPaused = false;
                let stepCount = 0;
                let countdownValue = refreshRate / 1000;
                let progressInterval;
                let refreshInterval;
                
                function setRefreshRate(rate) {
                    refreshRate = rate;
                    countdownValue = rate / 1000;
                    
                    // Update button states
                    document.querySelectorAll('.control-btn').forEach(btn => btn.classList.remove('active'));
                    if (rate === 1000) document.getElementById('fast').classList.add('active');
                    else if (rate === 2000) document.getElementById('normal').classList.add('active');
                    else if (rate === 3000) document.getElementById('slow').classList.add('active');
                    
                    if (!isPaused) {
                        restartRefresh();
                    }
                }
                
                function togglePause() {
                    isPaused = !isPaused;
                    const btn = document.getElementById('pauseBtn');
                    
                    if (isPaused) {
                        btn.textContent = 'Resume';
                        btn.classList.add('active');
                        clearInterval(refreshInterval);
                        clearInterval(progressInterval);
                        document.getElementById('status').textContent = '⏸️ Paused';
                        document.getElementById('status').className = 'connection-status disconnected';
                    } else {
                        btn.textContent = 'Pause';
                        btn.classList.remove('active');
                        restartRefresh();
                        document.getElementById('status').textContent = '🟢 Connected';
                        document.getElementById('status').className = 'connection-status connected';
                    }
                }
                
                function restartRefresh() {
                    clearInterval(refreshInterval);
                    clearInterval(progressInterval);
                    updateCountdown();
                    startProgressAnimation();
                    
                    refreshInterval = setInterval(() => {
                        if (!isPaused) {
                            refreshImage();
                            updateCountdown();
                            startProgressAnimation();
                        }
                    }, refreshRate);
                }
                
                function refreshImage() {
                    const img = document.getElementById('ecosystem');
                    const timestamp = new Date().getTime();
                    
                    // Add loading state
                    img.classList.add('img-loading');
                    
                    // Update image with cache-busting timestamp
                    img.src = `realtime_ecosystem.png?t=${timestamp}`;
                    stepCount++;
                    document.getElementById('step').textContent = stepCount;
                }
                
                function imageLoaded() {
                    const img = document.getElementById('ecosystem');
                    img.classList.remove('img-loading');
                }
                
                function updateCountdown() {
                    countdownValue = refreshRate / 1000;
                    document.getElementById('countdown').textContent = countdownValue;
                }
                
                function startProgressAnimation() {
                    const progressBar = document.getElementById('progressBar');
                    const countdownElement = document.getElementById('countdown');
                    let progress = 0;
                    let timeLeft = refreshRate / 1000;
                    
                    clearInterval(progressInterval);
                    progressBar.style.width = '0%';
                    
                    progressInterval = setInterval(() => {
                        if (isPaused) return;
                        
                        progress += 100 / (refreshRate / 100);
                        timeLeft -= 0.1;
                        
                        progressBar.style.width = Math.min(progress, 100) + '%';
                        countdownElement.textContent = Math.max(0, timeLeft).toFixed(1);
                        
                        if (progress >= 100) {
                            clearInterval(progressInterval);
                        }
                    }, 100);
                }
                
                // Initialize
                window.onload = function() {
                    restartRefresh();
                    refreshImage(); // Load first image immediately
                };
                
                // Handle page visibility changes
                document.addEventListener('visibilitychange', function() {
                    if (document.hidden) {
                        clearInterval(refreshInterval);
                        clearInterval(progressInterval);
                    } else if (!isPaused) {
                        restartRefresh();
                    }
                });
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        with open('ecosystem_display.html', 'w') as f:
            f.write(html_content)
        
        # Open web browser
        try:
            webbrowser.open(f'file://{os.path.abspath("ecosystem_display.html")}')
            print("✅ Opened web browser for real-time display")
        except:
            print("⚠️ Could not open browser automatically")
            print(f"📁 Open this file manually: {os.path.abspath('ecosystem_display.html')}")
        
        print("🔄 Starting real-time simulation...")
        print("   - Browser will auto-refresh every 2 seconds")
        print("   - Press Ctrl+C to stop")
        
        try:
            step = 0
            while step < 1000:  # Run for 1000 steps
                # Update simulation
                self.update_canvas()
                
                # Save current state as image
                plt.savefig('realtime_ecosystem.png', dpi=120, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                
                # Update every few steps for smoother display
                if step % 3 == 0:
                    print(f"   Step {step}: H={len([a for a in self.env.agents if hasattr(a, 'species_type') and a.species_type.value == 'herbivore'])}, C={len([a for a in self.env.agents if hasattr(a, 'species_type') and a.species_type.value == 'carnivore'])}")
                
                step += 1
                time.sleep(0.4)  # Update every 0.4 seconds for smooth simple view
                
        except KeyboardInterrupt:
            print("\n🛑 Web simulation stopped by user")
        
        print("🏁 Simulation complete!")
    
    def start_modern_web_display(self):
        """Start modern real-time web display with WebSocket streaming"""
        try:
            from .realtime_web_server import start_realtime_web_server
            print("🚀 Launching modern real-time web interface...")
            print("   - WebSocket streaming (no page refreshes)")
            print("   - Real-time canvas updates")
            print("   - Interactive controls")
            start_realtime_web_server(self)
        except ImportError as e:
            print(f"❌ Could not start web server: {e}")
            print("📦 Installing required packages...")
            import subprocess
            try:
                subprocess.check_call(['pip', 'install', 'flask', 'flask-socketio'])
                print("✅ Packages installed, restart the application")
            except:
                print("❌ Could not install packages automatically")
                print("💡 Run: pip install flask flask-socketio")
        except Exception as e:
            print(f"❌ Web server error: {e}")
            print("🔄 Falling back to traditional web display...")
            self.start_web_animation()
    
    def on_close(self, event):
        """Handle window close event"""
        self.running = False
        print("\n🏁 Canvas closed. Final statistics:")
        final_stats = self.env.get_neural_stats()
        print(f"   Steps completed: {self.step_count}")
        print(f"   Final herbivores: {final_stats['herbivore_count']}")
        print(f"   Final carnivores: {final_stats['carnivore_count']}")
        print(f"   Final avg fitness: {final_stats.get('avg_neural_fitness', 0):.1f}")
    
    def cleanup(self):
        """Clean up resources"""
        plt.ioff()
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()

class AdvancedEcosystemCanvas:
    """Enhanced version with additional visualizations"""
    
    def __init__(self, width=200, height=200):
        self.width = width
        self.height = height
        self.env = NeuralEnvironment(width=width, height=height, use_neural_agents=True)
        
        # Set up multi-panel figure
        plt.ion()
        self.fig = plt.figure(figsize=(16, 12))
        
        # Main ecosystem view
        self.ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_main.set_xlim(0, width)
        self.ax_main.set_ylim(0, height)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('Live Ecosystem', fontsize=14, fontweight='bold')
        
        # Fitness evolution plot
        self.ax_fitness = plt.subplot2grid((3, 3), (0, 2))
        self.ax_fitness.set_title('Fitness Evolution', fontsize=12)
        
        # Population dynamics
        self.ax_population = plt.subplot2grid((3, 3), (1, 2))
        self.ax_population.set_title('Population', fontsize=12)
        
        # Neural diversity visualization
        self.ax_neural = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        self.ax_neural.set_title('Neural Diversity Evolution', fontsize=12)
        
        # Data tracking
        self.fitness_history_herb = deque(maxlen=100)
        self.fitness_history_carn = deque(maxlen=100)
        self.population_history = deque(maxlen=100)
        self.neural_diversity_history = deque(maxlen=100)
        self.neural_signatures = deque(maxlen=50)  # Store neural network "fingerprints"
        self.step_count = 0
        self.running = True
        
        self.setup_advanced_canvas()
    
    def setup_advanced_canvas(self):
        """Set up the advanced multi-panel canvas"""
        # Main plot setup
        self.ax_main.set_facecolor('lightblue')
        self.ax_main.grid(True, alpha=0.2)
        
        # Fitness plot setup
        self.ax_fitness.set_ylabel('Fitness')
        self.ax_fitness.grid(True, alpha=0.3)
        
        # Population plot setup
        self.ax_population.set_ylabel('Count')
        self.ax_population.grid(True, alpha=0.3)
        
        # Neural diversity visualization setup
        self.ax_neural.set_facecolor('black')
        self.ax_neural.set_ylabel('Neural Signature')
        self.ax_neural.set_xlabel('Evolution Timeline')
        
        plt.tight_layout()
    
    def update_advanced_canvas(self):
        """Update all panels of the advanced canvas"""
        # Step simulation
        self.env.step()
        self.step_count += 1
        
        # Get data
        data = self.get_agent_data()
        stats = self.env.get_neural_stats()
        
        # Update main ecosystem view
        self.ax_main.clear()
        self.ax_main.set_xlim(0, self.width)
        self.ax_main.set_ylim(0, self.height)
        self.ax_main.set_title(f'Live Ecosystem (Step {self.step_count})', fontsize=14, fontweight='bold')
        self.ax_main.set_facecolor('lightblue')
        self.ax_main.grid(True, alpha=0.2)
        
        # Plot agents and food
        if data['food']['x']:
            self.ax_main.scatter(data['food']['x'], data['food']['y'], c='brown', s=40, marker='s', alpha=0.7)
        
        if data['herbivores']['x']:
            self.ax_main.scatter(data['herbivores']['x'], data['herbivores']['y'], 
                               c='green', s=60, marker='o', alpha=0.8, edgecolors='darkgreen')
        
        if data['carnivores']['x']:
            self.ax_main.scatter(data['carnivores']['x'], data['carnivores']['y'], 
                               c='red', s=80, marker='^', alpha=0.8, edgecolors='darkred')
        
        # Update fitness tracking
        herb_fitness = np.mean(data['herbivores']['fitness']) if data['herbivores']['fitness'] else 0
        carn_fitness = np.mean(data['carnivores']['fitness']) if data['carnivores']['fitness'] else 0
        
        self.fitness_history_herb.append(herb_fitness)
        self.fitness_history_carn.append(carn_fitness)
        
        # Update fitness plot
        self.ax_fitness.clear()
        self.ax_fitness.set_title('Fitness Evolution', fontsize=12)
        if len(self.fitness_history_herb) > 1:
            self.ax_fitness.plot(list(self.fitness_history_herb), 'g-', label='Herbivores', linewidth=2)
            self.ax_fitness.plot(list(self.fitness_history_carn), 'r-', label='Carnivores', linewidth=2)
            self.ax_fitness.legend()
        self.ax_fitness.grid(True, alpha=0.3)
        
        # Update population tracking
        self.population_history.append({
            'herbivores': len(data['herbivores']['x']),
            'carnivores': len(data['carnivores']['x']),
            'food': len(data['food']['x'])
        })
        
        # Update population plot
        self.ax_population.clear()
        self.ax_population.set_title('Population', fontsize=12)
        if len(self.population_history) > 1:
            herb_pop = [p['herbivores'] for p in self.population_history]
            carn_pop = [p['carnivores'] for p in self.population_history]
            food_pop = [p['food'] for p in self.population_history]
            
            self.ax_population.plot(herb_pop, 'g-', label='Herbivores', linewidth=2)
            self.ax_population.plot(carn_pop, 'r-', label='Carnivores', linewidth=2)
            self.ax_population.plot(food_pop, 'brown', linestyle='--', label='Food', alpha=0.7)
            self.ax_population.legend()
        self.ax_population.grid(True, alpha=0.3)
        
        # Update neural diversity visualization
        self.visualize_neural_diversity()
        
        plt.tight_layout()
        return True
    
    def get_agent_data(self):
        """Get agent data for advanced visualization"""
        neural_agents = [agent for agent in self.env.agents if isinstance(agent, NeuralAgent)]
        
        herbivores = [a for a in neural_agents if a.species_type == SpeciesType.HERBIVORE]
        carnivores = [a for a in neural_agents if a.species_type == SpeciesType.CARNIVORE]
        
        return {
            'herbivores': {
                'x': [a.position.x for a in herbivores],
                'y': [a.position.y for a in herbivores],
                'fitness': [a.brain.fitness_score for a in herbivores]
            },
            'carnivores': {
                'x': [a.position.x for a in carnivores],
                'y': [a.position.y for a in carnivores],
                'fitness': [a.brain.fitness_score for a in carnivores]
            },
            'food': {
                'x': [f.position.x for f in self.env.food_sources if f.is_available],
                'y': [f.position.y for f in self.env.food_sources if f.is_available]
            }
        }
    
    def extract_neural_signatures(self):
        """Extract neural network signatures for diversity analysis"""
        neural_agents = [agent for agent in self.env.agents if isinstance(agent, NeuralAgent)]
        
        signatures = []
        for agent in neural_agents:
            # Extract key neural network characteristics
            brain = agent.brain
            
            # Get weight patterns from first layer (simplified signature)
            if hasattr(brain, 'weights') and len(brain.weights) > 0:
                # Flatten first layer weights and take sample
                first_layer = brain.weights[0].flatten()
                
                # Create a neural signature using weight patterns
                signature = {
                    'species': agent.species_type.value,
                    'fitness': brain.fitness_score,
                    'weight_mean': np.mean(first_layer),
                    'weight_std': np.std(first_layer),
                    'weight_range': np.max(first_layer) - np.min(first_layer),
                    'activation_pattern': hash(tuple(first_layer[:8].round(2))),  # Simplified fingerprint
                    'decisions_made': brain.decisions_made,
                    'energy_level': agent.energy / agent.max_energy
                }
                signatures.append(signature)
        
        return signatures
    
    def visualize_neural_diversity(self):
        """Create beautiful neural diversity visualization"""
        self.ax_neural.clear()
        self.ax_neural.set_facecolor('white')
        self.ax_neural.set_title('Neural Diversity Evolution', fontsize=12, color='black')
        
        # Get current neural signatures
        current_signatures = self.extract_neural_signatures()
        
        if not current_signatures:
            return
        
        # Store signatures for timeline
        self.neural_signatures.append(current_signatures)
        
        # Calculate diversity metrics
        herbivore_sigs = [s for s in current_signatures if s['species'] == 'herbivore']
        carnivore_sigs = [s for s in current_signatures if s['species'] == 'carnivore']
        
        # Neural diversity score (based on weight pattern variety)
        if len(herbivore_sigs) > 1:
            herb_weights = [s['weight_mean'] for s in herbivore_sigs]
            herb_diversity = np.std(herb_weights) * len(set([s['activation_pattern'] for s in herbivore_sigs]))
        else:
            herb_diversity = 0
            
        if len(carnivore_sigs) > 1:
            carn_weights = [s['weight_mean'] for s in carnivore_sigs]
            carn_diversity = np.std(carn_weights) * len(set([s['activation_pattern'] for s in carnivore_sigs]))
        else:
            carn_diversity = 0
        
        total_diversity = herb_diversity + carn_diversity
        self.neural_diversity_history.append(total_diversity)
        
        # Create neural constellation plot
        if len(current_signatures) > 0:
            # Plot 1: Neural constellation (2D projection of neural space)
            x_positions = []
            y_positions = []
            colors = []
            sizes = []
            
            for i, sig in enumerate(current_signatures):
                # Map neural characteristics to 2D space
                x = sig['weight_mean'] * 10  # Spread out the visualization
                y = sig['weight_std'] * 10
                
                x_positions.append(x)
                y_positions.append(y)
                
                # Color by species and fitness
                if sig['species'] == 'herbivore':
                    color_base = np.array([0.2, 0.8, 0.2])  # Green base
                else:
                    color_base = np.array([0.8, 0.2, 0.2])  # Red base
                
                # Modulate brightness by fitness
                fitness_factor = min(1.0, sig['fitness'] / 20.0)
                color = color_base * (0.3 + 0.7 * fitness_factor)
                colors.append(color)
                
                # Size by energy level and decisions made
                size = 20 + (sig['energy_level'] * 30) + (min(sig['decisions_made'], 50) * 0.5)
                sizes.append(size)
            
            # Create the constellation plot
            scatter = self.ax_neural.scatter(x_positions, y_positions, c=colors, s=sizes, 
                                           alpha=0.7, edgecolors='white', linewidth=0.5)
            
            # Add neural diversity timeline as background
            if len(self.neural_diversity_history) > 1:
                # Normalize diversity history for background effect
                diversity_line = np.array(list(self.neural_diversity_history))
                diversity_line = (diversity_line - np.min(diversity_line)) / (np.max(diversity_line) - np.min(diversity_line) + 0.001)
                
                # Create flowing background based on diversity
                timeline_x = np.linspace(-2, 2, len(diversity_line))
                timeline_y = diversity_line * 2 - 1  # Center around 0
                
                # Add flowing energy lines
                for i in range(len(timeline_x)-1):
                    alpha = 0.3 * (i / len(timeline_x))  # Fade towards recent
                    self.ax_neural.plot([timeline_x[i], timeline_x[i+1]], 
                                      [timeline_y[i], timeline_y[i+1]], 
                                      color='cyan', alpha=alpha, linewidth=2)
            
            # Add connection lines between similar neural patterns
            for i, sig1 in enumerate(current_signatures):
                for j, sig2 in enumerate(current_signatures[i+1:], i+1):
                    # Calculate neural similarity
                    weight_diff = abs(sig1['weight_mean'] - sig2['weight_mean'])
                    if weight_diff < 0.1 and sig1['species'] == sig2['species']:  # Similar neural patterns
                        # Draw connection line
                        alpha = max(0.1, 0.5 - weight_diff * 5)
                        self.ax_neural.plot([x_positions[i], x_positions[j]], 
                                          [y_positions[i], y_positions[j]], 
                                          color='yellow', alpha=alpha, linewidth=1)
            
            # Style the plot
            self.ax_neural.set_xlim(-2, 2)
            self.ax_neural.set_ylim(-2, 2)
            self.ax_neural.set_xlabel('Neural Weight Patterns →', color='black')
            self.ax_neural.set_ylabel('Neural Complexity ↑', color='black')
            
            # Add diversity metrics as text
            diversity_text = f'Diversity Score: {total_diversity:.1f}\n'
            diversity_text += f'Herbivore Variants: {len(set([s["activation_pattern"] for s in herbivore_sigs]))}\n'
            diversity_text += f'Carnivore Variants: {len(set([s["activation_pattern"] for s in carnivore_sigs]))}'
            
            self.ax_neural.text(0.02, 0.98, diversity_text, transform=self.ax_neural.transAxes,
                              verticalalignment='top', horizontalalignment='left',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                              color='blue', fontsize=9)
            
            # Add legend
            legend_text = '🌟 Neural Constellation\n• Size = Energy + Experience\n• Color = Species + Fitness\n• Lines = Similar Patterns'
            self.ax_neural.text(0.98, 0.98, legend_text, transform=self.ax_neural.transAxes,
                              verticalalignment='top', horizontalalignment='right',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                              color='black', fontsize=8)
            
            # Remove axis ticks for cleaner look
            self.ax_neural.set_xticks([])
            self.ax_neural.set_yticks([])
    
    def start_advanced_animation(self):
        """Start the advanced multi-panel animation"""
        print("Starting Advanced Ecosystem Canvas")
        print("=" * 50)
        
        if WEB_MODE:
            print("🌐 Starting advanced modern web display")
            self.start_modern_web_display()
            return
        elif HEADLESS_MODE:
            print("📱 Running advanced canvas in headless mode")
            print("   - Multi-panel visualization")
            print("   - Saving snapshots every 10 steps")
            
            # Run simulation and save snapshots
            for step in range(100):
                self.update_advanced_canvas()
                
                # Save snapshot every 10 steps
                if step % 10 == 0:
                    filename = f'advanced_ecosystem_step_{step:03d}.png'
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"   Saved: {filename}")
            
            print("Advanced simulation complete! Check saved PNG files.")
            return
        
        print("🖥️ Starting advanced GUI window display")
        print("Multi-panel visualization:")
        print("   - Main view: Live ecosystem")
        print("   - Fitness evolution over time")
        print("   - Population dynamics")
        print("   - Neural decision activity")
        print("=" * 50)
        
        # Enable interactive plotting for GUI mode
        plt.ion()
        
        # Create animation with cache_frame_data=False to avoid warnings
        self.ani = animation.FuncAnimation(self.fig, lambda frame: self.update_advanced_canvas(), 
                                         interval=200, blit=False, repeat=True, cache_frame_data=False)
        
        self.fig.canvas.mpl_connect('close_event', lambda event: setattr(self, 'running', False))
        
        # Show the window
        plt.show()
        
        try:
            while self.running:
                plt.pause(0.1)
                # Force window update
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        except KeyboardInterrupt:
            print("\n🛑 Advanced canvas stopped")
        
        plt.ioff()
    
    def start_advanced_web_animation(self):
        """Start advanced web-based animation"""
        import tempfile
        import webbrowser
        import time
        
        print("🌐 Creating advanced web-based real-time display...")
        
    def start_advanced_web_animation(self):
        """Start advanced web-based animation"""
        import tempfile
        import webbrowser
        import time
        
        print("🌐 Creating advanced web-based real-time display...")
        
        # Create HTML for advanced display
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Ecosystem Advanced Real-Time Display</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    text-align: center; 
                    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); 
                    color: white; 
                    margin: 0; 
                    padding: 15px;
                    overflow-x: hidden;
                    min-height: 100vh;
                }
                .container { max-width: 1400px; margin: 0 auto; }
                img { 
                    max-width: 98%; 
                    height: auto; 
                    border: 3px solid #4a90e2; 
                    border-radius: 15px; 
                    box-shadow: 0 12px 40px rgba(74, 144, 226, 0.3);
                    transition: all 0.6s ease;
                    background: #2a2a3e;
                }
                .img-loading { 
                    opacity: 0.8; 
                    transform: scale(0.98);
                    filter: blur(1px);
                }
                .status { 
                    margin: 15px; 
                    padding: 15px; 
                    background: rgba(74, 144, 226, 0.1); 
                    border-radius: 12px; 
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(74, 144, 226, 0.3);
                }
                .title { 
                    color: #4a90e2; 
                    font-size: 32px; 
                    margin-bottom: 10px; 
                    text-shadow: 0 3px 6px rgba(0,0,0,0.5);
                    animation: glow 3s ease-in-out infinite alternate;
                }
                @keyframes glow {
                    from { text-shadow: 0 3px 6px rgba(74, 144, 226, 0.4); }
                    to { text-shadow: 0 3px 12px rgba(74, 144, 226, 0.8); }
                }
                .panels { 
                    color: #ffa726; 
                    font-size: 16px;
                    margin: 10px 0;
                }
                .controls {
                    margin: 20px;
                    display: flex;
                    justify-content: center;
                    gap: 12px;
                    flex-wrap: wrap;
                }
                .control-btn {
                    padding: 10px 18px;
                    background: linear-gradient(135deg, #4a90e2, #357abd);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                }
                .control-btn:hover { 
                    background: linear-gradient(135deg, #357abd, #2968a3);
                    transform: translateY(-2px);
                    box-shadow: 0 6px 16px rgba(0,0,0,0.3);
                }
                .control-btn.active { 
                    background: linear-gradient(135deg, #ff9800, #f57c00);
                    box-shadow: 0 4px 12px rgba(255, 152, 0, 0.4);
                }
                .progress-container {
                    width: 100%;
                    background: rgba(255,255,255,0.1);
                    border-radius: 20px;
                    overflow: hidden;
                    margin: 15px 0;
                    height: 6px;
                }
                .progress-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #4a90e2, #42a5f5, #66bb6a);
                    width: 0%;
                    transition: width 0.2s ease;
                    border-radius: 20px;
                }
                .analytics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }
                .analytics-card {
                    background: rgba(74, 144, 226, 0.08);
                    padding: 18px;
                    border-radius: 10px;
                    border: 1px solid rgba(74, 144, 226, 0.2);
                    backdrop-filter: blur(5px);
                }
                .analytics-card h3 {
                    margin: 0 0 10px 0;
                    color: #4a90e2;
                    font-size: 18px;
                }
                .connection-indicator {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 8px 15px;
                    border-radius: 25px;
                    font-size: 12px;
                    font-weight: bold;
                    z-index: 1000;
                    transition: all 0.3s ease;
                }
                .connected { 
                    background: linear-gradient(135deg, #4caf50, #66bb6a);
                    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
                }
                .disconnected { 
                    background: linear-gradient(135deg, #f44336, #ef5350);
                    box-shadow: 0 4px 12px rgba(244, 67, 54, 0.3);
                }
                .loading-spinner {
                    display: none;
                    margin: 10px auto;
                    width: 20px;
                    height: 20px;
                    border: 2px solid #4a90e2;
                    border-top: 2px solid transparent;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="connection-indicator connected" id="status">🟢 Live</div>
                
                <div class="title">🧬 AI Ecosystem - Advanced Multi-Panel Display</div>
                
                <div class="controls">
                    <button class="control-btn" onclick="setRefreshRate(1500)" id="fast">Fast (1.5s)</button>
                    <button class="control-btn active" onclick="setRefreshRate(2500)" id="normal">Normal (2.5s)</button>
                    <button class="control-btn" onclick="setRefreshRate(4000)" id="slow">Slow (4s)</button>
                    <button class="control-btn" onclick="togglePause()" id="pauseBtn">Pause</button>
                    <button class="control-btn" onclick="downloadImage()" id="saveBtn">📷 Save</button>
                </div>
                
                <div class="status">
                    <div class="progress-container">
                        <div class="progress-fill" id="progressBar"></div>
                    </div>
                    <p>Step: <span id="step">0</span> | Next update: <span id="countdown">2.5</span>s</p>
                    <div class="panels">
                        <p>📊 Live Ecosystem | 📈 Fitness Evolution | 📊 Population Dynamics | 🌟 Neural Diversity</p>
                    </div>
                    <div class="loading-spinner" id="spinner"></div>
                </div>
                
                <img id="ecosystem" src="realtime_advanced_ecosystem.png" alt="Advanced Ecosystem Display" onload="imageLoaded()" onerror="imageError()">
                
                <div class="analytics-grid">
                    <div class="analytics-card">
                        <h3>🦌 Herbivore Agents</h3>
                        <p>Neural networks learning to find food and avoid predators</p>
                        <p><strong>Green circles</strong> - size shows energy level</p>
                    </div>
                    <div class="analytics-card">
                        <h3>🐺 Carnivore Agents</h3>
                        <p>AI predators learning hunting strategies</p>
                        <p><strong>Red triangles</strong> - size shows energy level</p>
                    </div>
                    <div class="analytics-card">
                        <h3>📈 Fitness Evolution</h3>
                        <p>Real-time learning progress tracking</p>
                        <p>Watch AI agents become smarter over time</p>
                    </div>
                    <div class="analytics-card">
                        <h3>🌟 Neural Diversity</h3>
                        <p>Evolution of neural network patterns</p>
                        <p>Constellation view shows emerging intelligence variants</p>
                    </div>
                </div>
                
                <div class="status">
                    <p><strong>🚀 Advanced Multi-Panel Neural Learning Simulation</strong></p>
                    <p>Observing artificial intelligence emergence through evolutionary pressure!</p>
                </div>
            </div>
            
            <script>
                let refreshRate = 2500; // Default 2.5 seconds for advanced view
                let isPaused = false;
                let stepCount = 0;
                let countdownValue = refreshRate / 1000;
                let progressInterval;
                let refreshInterval;
                let loadingTimeout;
                
                function setRefreshRate(rate) {
                    refreshRate = rate;
                    countdownValue = rate / 1000;
                    
                    // Update button states
                    document.querySelectorAll('.control-btn').forEach(btn => btn.classList.remove('active'));
                    if (rate === 1500) document.getElementById('fast').classList.add('active');
                    else if (rate === 2500) document.getElementById('normal').classList.add('active');
                    else if (rate === 4000) document.getElementById('slow').classList.add('active');
                    
                    if (!isPaused) {
                        restartRefresh();
                    }
                }
                
                function togglePause() {
                    isPaused = !isPaused;
                    const btn = document.getElementById('pauseBtn');
                    
                    if (isPaused) {
                        btn.textContent = '▶️ Resume';
                        btn.classList.add('active');
                        clearInterval(refreshInterval);
                        clearInterval(progressInterval);
                        clearTimeout(loadingTimeout);
                        document.getElementById('status').textContent = '⏸️ Paused';
                        document.getElementById('status').className = 'connection-indicator disconnected';
                        document.getElementById('spinner').style.display = 'none';
                    } else {
                        btn.textContent = 'Pause';
                        btn.classList.remove('active');
                        restartRefresh();
                        document.getElementById('status').textContent = '🟢 Live';
                        document.getElementById('status').className = 'connection-indicator connected';
                    }
                }
                
                function downloadImage() {
                    const img = document.getElementById('ecosystem');
                    const link = document.createElement('a');
                    link.download = `ecosystem_step_${stepCount}.png`;
                    link.href = img.src;
                    link.click();
                }
                
                function restartRefresh() {
                    clearInterval(refreshInterval);
                    clearInterval(progressInterval);
                    clearTimeout(loadingTimeout);
                    updateCountdown();
                    startProgressAnimation();
                    
                    refreshInterval = setInterval(() => {
                        if (!isPaused) {
                            refreshImage();
                            updateCountdown();
                            startProgressAnimation();
                        }
                    }, refreshRate);
                }
                
                function refreshImage() {
                    const img = document.getElementById('ecosystem');
                    const spinner = document.getElementById('spinner');
                    const timestamp = new Date().getTime();
                    
                    // Show loading state
                    img.classList.add('img-loading');
                    spinner.style.display = 'block';
                    
                    // Set timeout for loading indicator
                    loadingTimeout = setTimeout(() => {
                        spinner.style.display = 'none';
                    }, 1000);
                    
                    // Update image with cache-busting timestamp
                    img.src = `realtime_advanced_ecosystem.png?t=${timestamp}`;
                    stepCount++;
                    document.getElementById('step').textContent = stepCount;
                }
                
                function imageLoaded() {
                    const img = document.getElementById('ecosystem');
                    const spinner = document.getElementById('spinner');
                    img.classList.remove('img-loading');
                    spinner.style.display = 'none';
                    clearTimeout(loadingTimeout);
                }
                
                function imageError() {
                    const spinner = document.getElementById('spinner');
                    spinner.style.display = 'none';
                    console.log('Image loading error - simulation may be starting up');
                }
                
                function updateCountdown() {
                    countdownValue = refreshRate / 1000;
                    document.getElementById('countdown').textContent = countdownValue.toFixed(1);
                }
                
                function startProgressAnimation() {
                    const progressBar = document.getElementById('progressBar');
                    const countdownElement = document.getElementById('countdown');
                    let progress = 0;
                    let timeLeft = refreshRate / 1000;
                    
                    clearInterval(progressInterval);
                    progressBar.style.width = '0%';
                    
                    progressInterval = setInterval(() => {
                        if (isPaused) return;
                        
                        progress += 100 / (refreshRate / 100);
                        timeLeft -= 0.1;
                        
                        progressBar.style.width = Math.min(progress, 100) + '%';
                        countdownElement.textContent = Math.max(0, timeLeft).toFixed(1);
                        
                        if (progress >= 100) {
                            clearInterval(progressInterval);
                        }
                    }, 100);
                }
                
                // Initialize
                window.onload = function() {
                    setTimeout(() => {
                        restartRefresh();
                        refreshImage(); // Load first image after slight delay
                    }, 500);
                };
                
                // Handle page visibility changes
                document.addEventListener('visibilitychange', function() {
                    if (document.hidden) {
                        clearInterval(refreshInterval);
                        clearInterval(progressInterval);
                        clearTimeout(loadingTimeout);
                    } else if (!isPaused) {
                        setTimeout(restartRefresh, 300);
                    }
                });
                
                // Preload next image for smoother transitions
                function preloadNextImage() {
                    const preloadImg = new Image();
                    preloadImg.src = `realtime_advanced_ecosystem.png?t=${new Date().getTime()}`;
                }
                
                // Preload every few seconds
                setInterval(preloadNextImage, refreshRate - 200);
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        with open('advanced_ecosystem_display.html', 'w') as f:
            f.write(html_content)
        
        # Open web browser
        try:
            webbrowser.open(f'file://{os.path.abspath("advanced_ecosystem_display.html")}')
            print("✅ Opened web browser for advanced real-time display")
        except:
            print("⚠️ Could not open browser automatically")
            print(f"📁 Open this file manually: {os.path.abspath('advanced_ecosystem_display.html')}")
        
        print("🔄 Starting advanced real-time simulation...")
        print("   - Browser will auto-refresh every 3 seconds")
        print("   - Multi-panel view with detailed analytics")
        print("   - Press Ctrl+C to stop")
        
        try:
            step = 0
            while step < 1000:  # Run for 1000 steps
                # Update simulation
                self.update_advanced_canvas()
                
                # Save current state as image
                plt.savefig('realtime_advanced_ecosystem.png', dpi=120, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                
                # Status update every 5 steps
                if step % 5 == 0:
                    stats = self.env.get_neural_stats()
                    print(f"   Step {step}: H={stats.get('herbivore_count', 0)}, C={stats.get('carnivore_count', 0)}, Fitness={stats.get('avg_neural_fitness', 0):.1f}")
                
                step += 1
                time.sleep(0.5)  # Update every 0.5 seconds for smooth web experience
                
        except KeyboardInterrupt:
            print("\n🛑 Advanced web simulation stopped by user")
        
        print("🏁 Advanced simulation complete!")
    
    def start_modern_web_display(self):
        """Start modern real-time web display with WebSocket streaming"""
        try:
            from .realtime_web_server import start_realtime_web_server
            print("🚀 Launching advanced modern web interface...")
            print("   - WebSocket streaming (no page refreshes)")
            print("   - Multi-panel real-time updates")
            print("   - Interactive controls and charts")
            start_realtime_web_server(self)
        except ImportError as e:
            print(f"❌ Could not start web server: {e}")
            print("📦 Installing required packages...")
            import subprocess
            try:
                subprocess.check_call(['pip', 'install', 'flask', 'flask-socketio'])
                print("✅ Packages installed, restart the application")
            except:
                print("❌ Could not install packages automatically")
                print("💡 Run: pip install flask flask-socketio")
        except Exception as e:
            print(f"❌ Web server error: {e}")
            print("🔄 Falling back to traditional web display...")
            self.start_advanced_web_animation()

def run_realtime_canvas(canvas_type="simple"):
    """Run the real-time ecosystem canvas"""
    if canvas_type == "simple":
        canvas = RealTimeEcosystemCanvas(width=200, height=200, update_interval=150)
        canvas.start_animation()
    elif canvas_type == "advanced":
        canvas = AdvancedEcosystemCanvas(width=200, height=200)
        canvas.start_advanced_animation()

if __name__ == "__main__":
    print("🎨 Real-Time Ecosystem Canvas")
    print("=" * 40)
    print("1. Simple Canvas (single view)")
    print("2. Advanced Canvas (multi-panel)")
    print("3. Demo with Animation Save")
    
    choice = input("\nSelect canvas type (1-3): ").strip()
    
    if choice == "1":
        run_realtime_canvas("simple")
    elif choice == "2":
        run_realtime_canvas("advanced")
    elif choice == "3":
        canvas = RealTimeEcosystemCanvas(width=150, height=150, update_interval=100)
        canvas.start_animation(save_animation=True)
    else:
        print("Running simple canvas...")
        run_realtime_canvas("simple")
