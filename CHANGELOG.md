# Changelog

All notable changes to the Neural Ecosystem Simulation project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.3] - 2025-07-31

### 🧠 Enhanced Neural Network Food-Seeking

This update dramatically improves neural network sensory inputs and evolutionary learning for more effective food-seeking behavior.

### Added
- **🎯 Unit Vector Directional Inputs**
  - Clear directional signals pointing toward food sources (inputs 3-4)
  - Perfect unit vector magnitude (1.0) for precise direction indication
  - Separate directional vectors for threats/prey (inputs 6-7)
  - Eliminates directional precision loss from clamping

- **📈 Movement Alignment Fitness**
  - Rewards agents for moving toward food sources (+25 points max)
  - Penalties for moving away from food (-10 points max)
  - Real-time movement direction tracking and analysis
  - Promotes evolutionary learning of food-seeking behavior

- **🔒 Mathematical Safety Systems**
  - Infinite value protection in fitness calculations
  - NaN detection and prevention in neural network inputs
  - Capped bonus values to prevent mathematical overflow
  - Robust error handling for edge cases

### Improved
- **🧠 Neural Network Architecture**
  - Enhanced sensory input processing with clear food direction vectors
  - Improved threat/prey detection with unit vector precision
  - Better boundary awareness signals for natural edge avoidance
  - Consistent input normalization across all sensory channels

- **🧬 Evolutionary Performance**
  - Demonstrable improvement across generations: +26% average fitness
  - Food consumption increases: 5 → 9 → 13 food sources over 3 generations
  - Movement alignment improves from 44% → 65% agents moving toward food
  - Population growth: 28 → 39 → 48 agents through successful evolution

### Performance
- **Evolution Learning**: Neural networks learn food-seeking through natural selection
- **Fitness Growth**: 45.32 → 57.29 average fitness (+26% improvement)
- **Food Efficiency**: More agents successfully consuming food each generation
- **Direction Accuracy**: Perfect unit vectors (magnitude = 1.000) for movement guidance

### Technical Details
- **Unit Vector Calculation**: `direction = (dx, dy) / sqrt(dx² + dy²)`
- **Movement Alignment**: `alignment = food_direction · agent_movement`
- **Safety Checks**: `math.isinf()` and `math.isnan()` validation
- **Fitness Momentum**: Stability through weighted fitness history

## [3.0.2] - 2025-07-31

### 🍃 Food-Focused Fitness System - Natural Selection

This update replaces artificial boundary penalties with a biologically realistic fitness system focused on food acquisition, the core survival skill.

### Changed
- **🎯 Natural Fitness System**
  - Replaced boundary penalties with food-acquisition focused fitness
  - Primary reward for proximity to food sources (+15 points)
  - Major bonus for successful food consumption (+50 points)
  - Enhanced species-specific food efficiency bonuses

- **🌱 Enhanced Herbivore Fitness**
  - Food consumption bonus increased from 2x to 5x per food consumed
  - Food efficiency calculation: energy ratio / time spent foraging
  - Proximity rewards for being near available food sources
  - Movement tracking: bonus for approaching food (+10 points)

- **🦁 Enhanced Carnivore Fitness**
  - Hunt success bonus increased from 20x to 25x per successful hunt
  - Hunt efficiency calculation: energy gained relative to time
  - Prey proximity rewards for being near potential targets
  - Enhanced hunting range bonuses for strategic positioning

- **🚫 Removed Artificial Penalties**
  - Eliminated boundary fitness penalties (-50/-20 edge penalties)
  - Removed center-seeking bonuses (+10/+5 center rewards)
  - Natural boundary avoidance now emerges from food distribution

### Performance
- **Higher Fitness Scores**: 21-83 fitness range (vs previous 2-28 range)
- **Natural Behavior**: Agents evolve realistic foraging strategies
- **Emergent Boundary Avoidance**: Naturally avoid edges to stay near food
- **Population Stability**: 64+ agents with healthy growth patterns

### Technical Details
- **Food Proximity Algorithm**: Inverse distance calculation from nearest food
- **Consumption Tracking**: Real-time step-based food acquisition monitoring
- **Efficiency Metrics**: Energy-to-time ratios for both species
- **Movement Rewards**: Progressive bonuses for approaching food sources

## [3.0.1] - 2025-07-31

### 🔧 Critical Behavioral Fixes - Ecosystem Stability

This patch release addresses critical post-v3.0.0 ecosystem behavioral issues that were causing carnivore extinction and herbivore boundary-seeking behavior.

### Fixed
- **🐛 Boundary Behavior Issues**
  - Fixed inverted boundary input signals (0=edge, 1=center) in neural networks
  - Agents now properly avoid boundaries instead of seeking them
  - Eliminated edge-clustering behavior that caused starvation

- **⚖️ Fitness System Improvements**  
  - Added strong boundary fitness penalties (-50 for edge proximity)
  - Added center-seeking bonus (+10 for staying in safe zones)
  - Enhanced carnivore survival bonuses (+20 hunt reward, +8 energy bonus)

- **🍃 Enhanced Food Distribution**
  - Implemented clustered food placement in safe zones (30-70% of environment)
  - Food sources now avoid boundary areas to prevent edge-seeking
  - Slower food regeneration (120 steps) for better resource management

- **🦁 Improved Hunting Mechanics**
  - Increased hunt range from 3.0 to 5.0 units
  - Enhanced hunt success calculation with multiple factors
  - Improved energy transfer efficiency (70% → 80%)
  - Added health-based hunting bonuses for weak prey

### Performance
- **Population Stability**: 100% generation survival rate (vs previous crashes)
- **Growth Recovery**: Populations now grow from 27 to 111+ agents
- **Fitness Improvement**: Average fitness reaching 28.0+ (vs previous negatives)
- **Ecosystem Balance**: Stable carnivore-herbivore dynamics without extinction

### Technical Details
- **Boundary Input Fix**: Corrected neural network boundary distance calculation
- **Clustered Food Algorithm**: Food placement algorithm avoiding edges and clustering in center
- **Hunt Success Formula**: Multi-factor success rate based on energy, size, and health
- **Fitness Momentum**: Enhanced boundary penalties integrated with existing fitness system

## [3.0.0] - 2025-07-31

### 🧬 Pure Evolution System - Major Performance Optimization

This major release transitions to a pure evolution system after comprehensive analysis showing evolution significantly outperforms online learning approaches.

### Added
- **🚀 Enhanced Pure Evolution System**
  - Optimized genetic algorithm with 20% elite preservation (increased from 15%)
  - Tournament selection with size 4 for stronger selection pressure
  - Balanced mutation rates (15%) and high crossover rates (80%)
  - Advanced diversity bonuses to prevent population homogeneity
  - Comprehensive evolution tracking and visualization

- **📊 Evolution Analysis Tools**
  - `analyze_evolution_fitness.py` - Detailed fitness algorithm breakdown
  - `pure_evolution_demo.py` - Optimized pure evolution demonstration
  - `main_evolution.py` - Enhanced main system focused on evolution
  - Comprehensive performance comparison showing evolution outperforms learning by 50%+

- **🎯 Optimized Fitness Algorithm**
  - Survival fitness: +0.1 per step alive
  - Energy management: Up to +10 points based on health ratio
  - Species bonuses: +2 per food (herbivores), +15 per hunt (carnivores)
  - Reproduction rewards: +25 per offspring
  - Momentum smoothing: 90% previous + 10% current for stability

### Changed
- **System Architecture**: Transitioned from hybrid evolution+learning to pure evolution
- **Performance Focus**: Optimized for stability and long-term fitness improvement
- **Population Management**: Improved population growth control and genetic diversity
- **Configuration**: Streamlined configuration for optimal evolution parameters

### Removed
- **🧹 Obsolete Learning Files** - Removed underperforming learning systems:
  - `src/neural/online_learning.py` - Original online learning system
  - `src/neural/improved_online_learning.py` - Improved learning (still underperformed)
  - `src/neural/learning_agents.py` - Learning agent wrappers
  - `demos/enhanced_learning_demo.py` - Learning comparison demos
  - `demos/improved_learning_demo.py` - Improved learning demos

### Performance
- **66.7 fitness** (Evolution) vs **32.9 fitness** (Online Learning) - 103% improvement
- **Stable population management** with natural growth/decline cycles
- **Consistent genetic diversity** maintained across generations
- **Elite preservation** ensures best genetics persist across generations

### Technical Details
- Evolution uses proven biological principles: selection pressure, elite preservation, diversity maintenance
- Learning systems suffered from volatile real-time rewards, harsh boundary penalties, and weight instability
- Pure evolution provides stable fitness accumulation vs. volatile learning feedback
- Population-level optimization vs. individual learning optimization

## [2.1.0] - 2025-07-30

### 🧠 Enhanced Neural Network Inspection

This release significantly improves the neural network inspection capabilities with real-time visualization and advanced decision-making insights.

### Added
- **🎯 Real-Time Agent Inspection**
  - Click any agent in the simulation to view detailed neural network analysis
  - Live-updating agent information (energy, age, fitness, position)
  - Real-time neural network state visualization with D3.js
  - Dynamic field highlighting when values change

- **🔬 Advanced Neural Network Visualization**
  - Real-time sensory input display with 10-input neural network support
  - Live neural output visualization with decision strength indicators
  - Hidden layer activation visualization with color-coded neuron states
  - Connection strength visualization based on weights and activations
  - Strongest decision highlighting to show agent's primary choice

- **⚡ Enhanced Real-Time Updates**
  - Automatic modal updates when simulation is running
  - Smooth transition animations for changing values
  - Visual feedback for updated fields with highlight effects
  - Continuous neural network state monitoring

- **🎨 Improved Visual Design**
  - CSS animations for neural network activity (pulsing neurons)
  - Color-coded neural connections (green = positive, red = negative)
  - Activation-based opacity for neural connections
  - Professional styling with smooth transitions

### Fixed
- **🐛 Critical Agent Inspection Issues**
  - Fixed agent ID resolution (`agent.id` vs `agent.agent_id` mismatch)
  - Resolved neural network access errors (brain IS the neural network)
  - Fixed configuration attribute access (`config.input_size` direct access)
  - Corrected 10-input neural network support (was defaulting to 8)

- **🔧 Neural Network Data Processing**
  - Proper sensory input extraction from SensorSystem
  - Real hidden layer activation calculation
  - Accurate neural network output interpretation
  - Enhanced error handling for missing neural data

### Changed
- **📊 Input Label Updates**
  - Updated sensory input labels to reflect 10-input architecture:
    - Energy Level, Age Factor, Food Distance/Angle
    - Threat/Prey Distance/Angle, Population Density
    - Can Reproduce, X/Y Boundary Distance
  - Improved output labels: Move X/Y, Reproduce, Intensity

- **🎮 WebSocket Communication**
  - Added `update_agent` event for real-time agent updates
  - Enhanced agent inspection with live data streaming
  - Improved error handling for non-existent agents

### Technical Details

#### **Neural Network Architecture Support**
- **10→12→4 Architecture**: Full support for boundary-aware neural networks
- **Real-time Processing**: Live calculation of hidden layer activations
- **Decision Visualization**: Visual indication of strongest neural outputs
- **Activation Mapping**: Color-coded neurons based on activation strength

#### **WebSocket Events**
- **`inspect_agent`**: Initial agent inspection request
- **`update_agent`**: Real-time agent data updates
- **`agent_details`**: Complete agent information response
- **`agent_update`**: Live agent state updates
- **`agent_not_found`**: Graceful handling of removed agents

#### **JavaScript Enhancements**
- **Modal Management**: Smooth opening/closing with real-time updates
- **Field Highlighting**: Visual feedback for changing values
- **D3.js Integration**: Advanced neural network visualization
- **Animation System**: CSS-based smooth transitions

### Performance Improvements
- **Efficient Updates**: Only update open modals to reduce CPU usage
- **Smart Caching**: Reuse neural network calculations where possible
- **Optimized Rendering**: Efficient D3.js redrawing with selective updates
- **Memory Management**: Proper cleanup of event listeners and animations

---

## [2.0.0] - 2025-07-30

### 🎉 Major Release: Web Interface Overhaul

This release represents a complete restructuring of the visualization system, moving from multiple disparate display methods to a unified, modern web-based interface.

### Added
- **🌐 New Web Server Architecture**
  - `src/visualization/web_server.py` - Clean, simple web server with WebSocket support
  - `src/visualization/advanced_web_server.py` - Feature-rich web interface with neural network inspection
  - Real-time ecosystem visualization using HTML5 Canvas
  - Interactive controls for simulation start/stop and speed adjustment
  - Live population statistics and metrics dashboard
  - WebSocket-based real-time communication for seamless updates

- **🧠 Advanced Neural Network Features**
  - Interactive neural network inspection (click agents to view their brain structure)
  - D3.js-powered neural network visualizations
  - Real-time fitness tracking and generation evolution
  - Neural diversity metrics and analysis

- **🎮 Enhanced User Experience**
  - Modern responsive web interface design
  - Mobile-friendly layout and controls
  - Real-time activity log with timestamped events
  - Professional gradient backgrounds and smooth animations
  - Connection status indicators and error handling

- **🔧 Testing and Debugging**
  - `test_simple_websocket.py` - WebSocket connectivity testing tool
  - Enhanced error handling and fallback mechanisms
  - Improved compatibility between `step()` and `update()` methods

### Changed
- **📁 Major File Reorganization**
  - Updated `main.py` to use new web server architecture
  - Modified `README.md` with new web interface documentation
  - Updated `CONTRIBUTING.md` with current project structure

- **🔄 Simulation Loop Improvements**
  - Fixed compatibility issues between NeuralEnvironment and web server
  - Enhanced `SimpleEcosystemWrapper` with proper method detection
  - Improved error handling in simulation threads

### Removed
- **🗑️ Legacy Visualization System Cleanup**
  - `advanced_ecosystem_display.html` - Replaced by integrated web server
  - `launch_canvas.py` - Functionality integrated into main.py
  - `examples/main.py` - Consolidated into main entry point
  
- **🗑️ Deprecated Visualization Components**
  - `src/visualization/realtime_canvas.py` - Replaced by web_server.py
  - `src/visualization/realtime_web_server.py` - Replaced by advanced_web_server.py
  - `src/visualization/terminal_canvas.py` - Terminal display deprecated
  - `src/visualization/visualizer.py` - Functionality moved to web interface

- **🗑️ Outdated Testing Files**
  - `tests/test_generation_enhanced.py` - Superseded by current testing framework
  - `tests/test_generation_tracking.py` - Functionality tested elsewhere

- **🗑️ Configuration Files**
  - Previous `CHANGELOG.md` - Rebuilt from scratch for v2.0.0

### Fixed
- **🐛 Critical WebSocket Issues**
  - Resolved `'NeuralEnvironment' object has no attribute 'update'` error
  - Fixed threading conflicts in Flask-SocketIO configuration
  - Improved WebSocket connectivity and transport fallbacks
  - Enhanced error handling for method compatibility

- **🔧 Simulation Stability**
  - Fixed race conditions in simulation loop
  - Improved memory management for long-running simulations
  - Enhanced graceful shutdown handling

### Technical Details

#### **Web Server Architecture**
- **Framework**: Flask + Flask-SocketIO for real-time communication
- **Frontend**: HTML5 Canvas + JavaScript + WebSocket client
- **Visualization**: D3.js for neural networks, Chart.js for metrics
- **Threading**: Background simulation loop with WebSocket broadcasting

#### **Compatibility Improvements**
- **Method Detection**: Automatic detection of `step()` vs `update()` methods
- **Environment Wrapper**: Enhanced SimpleEcosystemWrapper with fallback mechanisms
- **Error Recovery**: Graceful handling of environment method mismatches

#### **Performance Enhancements**
- **Real-time Updates**: Optimized WebSocket message frequency
- **Canvas Rendering**: Efficient coordinate scaling and drawing routines
- **Memory Usage**: Improved data structure handling for large populations

### Migration Guide

#### **For Developers**
If you were using the old visualization system:

1. **Replace old imports**:
   ```python
   # Old
   from src.visualization.realtime_web_server import RealtimeEcosystemWebServer
   
   # New
   from src.visualization.web_server import EcosystemWebServer
   ```

2. **Update simulation launching**:
   ```python
   # Old
   python launch_canvas.py
   
   # New
   python main.py --web
   ```

3. **New web interface access**:
   - Open browser to `http://localhost:5000`
   - Use interactive controls instead of terminal commands

#### **For Users**
- **Start simulation**: `python main.py --web`
- **Access interface**: Open `http://localhost:5000` in your browser
- **Interactive controls**: Use web buttons instead of keyboard shortcuts
- **Real-time viewing**: No need to refresh - updates happen automatically

### Breaking Changes
- **Removed terminal-based visualization** - All visualization now web-based
- **Changed command-line interface** - Use `--web` flag for web interface
- **Deprecated standalone HTML files** - All HTML now served by Python server
- **Removed legacy examples** - Use main.py entry point instead

### Dependencies
- **Flask** - Web framework for serving interface
- **Flask-SocketIO** - Real-time WebSocket communication
- **No new external dependencies** - Uses existing NumPy, Matplotlib ecosystem

---

## Previous Versions

### [1.x.x] - Pre-2025
- Initial neural ecosystem implementation
- Terminal-based visualization system
- Basic genetic algorithms and neural networks
- Multiple disparate visualization approaches

---

## Roadmap

### Planned for v2.1.0
- **Enhanced Neural Inspection**: More detailed neural network analysis
- **Export Features**: Save/load evolved populations
- **Advanced Metrics**: Evolutionary tree visualization
- **Multi-Environment**: Connected ecosystem simulations

### Planned for v3.0.0
- **Machine Learning Integration**: TensorFlow/PyTorch backends
- **Advanced Genetics**: Sexual reproduction systems
- **Environmental Challenges**: Dynamic environmental conditions
- **Performance Optimization**: GPU acceleration support

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## Support

- **Issues**: Report bugs and request features on GitHub
- **Documentation**: Check README.md for usage instructions
- **Web Interface**: Access the live simulation at `http://localhost:5000`
