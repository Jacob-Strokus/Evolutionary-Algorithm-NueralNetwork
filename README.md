# 🧠 AI Neural Ecosystem Simulation

An advanced ecosystem simulation featuring neural network-powered agents that learn and evolve in real-time through genetic algorithms and sophisticated web-based visualization.

## ✨ Features

🧠 **Advanced Neural Intelligence**: Agents use 10→12→4 feedforward neural networks with boundary-aware sensory systems  
🧬 **Genetic Evolution**: Sophisticated genetic algorithms with crossover, mutation, and fitness-based selection  
🎮 **Real-time Web Interface**: Interactive web-based visualization with live agent inspection and neural network analysis  
📊 **Live Neural Inspection**: Click any agent to view real-time neural network activity with D3.js visualization  
🌐 **Boundary Awareness**: Neural networks include environmental boundary detection to prevent clustering  
📈 **Advanced Analytics**: Multi-panel charts tracking fitness evolution, population dynamics, and neural diversity  
🔍 **Agent Inspector**: Detailed neural network analysis with hidden layer activation visualization  
🎨 **Modern Design**: Professional color scheme with smooth animations and responsive interface  
⚡ **Real-time Updates**: Live-streaming agent data with WebSocket communication  

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd EA-NN

# Install dependencies
pip install -r requirements.txt

# Run the advanced web interface (recommended)
python main.py --web
# Then open: http://localhost:5000

# Or run neural simulation examples
python examples/main_neural.py
```

## 🎯 Web Interface Features

### �️ Real-time Ecosystem Display
- **Live Simulation**: Smooth real-time agent movement without page refresh
- **Interactive Canvas**: Click agents for instant neural network inspection
- **Speed Controls**: Adjust simulation speed from 10ms to 500ms per step
- **Population Metrics**: Live charts showing energy, fitness, and generation data
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile

### 🧠 Neural Network Inspector
- **One-Click Analysis**: Click any agent to open detailed neural inspection modal
- **Real-time Updates**: Agent data updates live while simulation runs
- **10-Input Visualization**: Complete sensory input display with current values
- **Hidden Layer Activity**: Real-time visualization of 12 hidden neurons with activation levels
- **Decision Strength**: Visual indication of strongest neural outputs
- **Modern Color Scheme**: Professional purple/teal and blue/orange color palette
- **D3.js Powered**: Smooth animations and interactive neural network diagrams

### 📊 Advanced Metrics Dashboard
- **Generation Tracking**: Complete evolutionary lineage from Generation 1+
- **Fitness Evolution**: Real-time fitness progression charts
- **Population Dynamics**: Live agent count and energy distribution
- **Neural Diversity**: Visualization of genetic diversity across generations  

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd coolshit

# Install dependencies
pip install -r requirements.txt

# Run neural simulation with web interface
python examples/main_neural.py
# Choose option 1 for basic simulation or 4 for extended analysis

# Or run the main interactive menu
python main.py
```

## 🏗️ Project Structure

```
src/
├── core/              # Core ecosystem mechanics and base classes
├── neural/            # Neural network agents and decision-making systems
├── evolution/         # Genetic algorithms and evolutionary operations
├── visualization/     # Real-time web server with neural inspection
│   ├── web_server.py           # Advanced web interface with neural inspector
│   ├── advanced_web_server.py  # Feature-rich visualization server
│   └── neural_visualizer.py    # D3.js neural network rendering
└── analysis/          # Data analysis and neural network inspection tools

examples/              # Ready-to-run simulation examples
├── main_neural.py     # Primary neural simulation interface
└── neural_learning_demo.py    # Neural learning demonstrations

tests/                 # Comprehensive test suite with neural network validation
scripts/               # Utility scripts, debugging tools, and training helpers
docs/                  # Phase completion reports and technical documentation
```

## 🎯 Usage Examples

### 🌟 Advanced Web Interface (Recommended)
```bash
python main.py --web
# Open browser to: http://localhost:5000
```

**Key Features:**
- **🖱️ Interactive Agent Inspection**: Click any agent for instant neural analysis
- **⚡ Real-time Updates**: Live data streaming with WebSocket communication
- **🧠 Neural Network Visualization**: D3.js-powered neural diagrams with:
  - Purple/teal neurons for positive/negative activations
  - Blue/orange connections for positive/negative weights
  - Real-time hidden layer activation display
  - Decision strength indicators
- **📊 Live Metrics**: Real-time population, energy, and generation tracking
- **🎮 Interactive Controls**: Simulation start/stop and speed adjustment
- **📱 Responsive Design**: Works on all devices with modern browsers

### Neural Ecosystem Simulation
```bash
python examples/main_neural.py
# Choose simulation type:
# 1. Basic Neural Network Simulation (500 steps)
# 2. Traditional vs Neural Comparison
# 3. Neural Learning Demo
# 4. Extended Neural Simulation (1500 steps)
```

### Traditional Ecosystem
```bash
python main.py
# Multi-panel traditional ecosystem simulation
```

## 🧬 Neural Architecture

### 🧠 Advanced Neural Intelligence
- **10→12→4 Architecture**: Enhanced neural network with expanded sensory inputs
- **10-Input Sensory System**: 
  - Energy Level, Age Factor
  - Food Distance & Angle detection
  - Threat/Prey Distance & Angle awareness
  - Population Density sensing
  - Reproduction Readiness state
  - Boundary Distance (X/Y) for environmental awareness
- **12-Neuron Hidden Layer**: Complex decision processing with activation visualization
- **4-Output Actions**: Movement (X/Y), reproduction attempt, and intensity modulation
- **Real-time Processing**: Live calculation of hidden layer activations for inspection

### 🎨 Visual Neural Network Features
- **Activation Visualization**: Real-time neuron activity with color-coded intensity
- **Connection Strength**: Visual weight representation with blue/orange color scheme
- **Decision Indicators**: Strongest output highlighting for agent behavior analysis
- **Hidden Layer Insights**: Live monitoring of internal neural processing
- **Professional Design**: Modern purple/teal and blue/orange color palette

### 🧬 Evolutionary Mechanics
- **Generation Tracking**: Complete lineage tracking from Generation 1 onwards
- **Genetic Operations**: Advanced crossover between compatible agents with neural weight inheritance
- **Mutation System**: Configurable mutation rates for neural network evolution
- **Fitness Selection**: Survival-based selection pressure with energy efficiency metrics
- **Population Dynamics**: Dynamic population management with reproduction cooldowns
- **Neural Inheritance**: Sophisticated neural network crossover and mutation algorithms

## 🌟 Key Innovations

### 🎯 Real-time Neural Inspection
Revolutionary agent analysis system:
- **One-Click Access**: Click any agent in the simulation for instant neural analysis
- **Live Updates**: Agent data streams in real-time while simulation runs
- **Visual Feedback**: Smooth animations highlight changing values
- **Neural Visualization**: Complete neural network structure with D3.js rendering
- **Decision Analysis**: Visual indication of strongest neural outputs and decision-making

### 🎨 Modern Color Scheme Design
Professional visual design for enhanced user experience:
- **Neural Connections**: Blue (#2196F3) for positive weights, Orange (#FF9800) for negative
- **Neuron Activations**: Purple (rgba(156, 39, 176)) for positive, Teal (rgba(0, 150, 136)) for negative
- **Inactive States**: Elegant gray (#e0e0e0) for inactive neurons
- **Consistent Theme**: Cohesive color palette across all neural visualization elements
- **Accessibility**: High contrast ratios for improved readability

### 🔧 Boundary Clustering Solution
Critical ecosystem behavior enhancement:
- **Problem Identified**: Neural networks lacked boundary awareness, causing agents to cluster at edges
- **Solution Implemented**: Added boundary distance inputs [8] and [9] to the neural sensory system
- **Result Achieved**: Agents now naturally avoid boundaries and explore the full environment
- **Performance Impact**: Improved agent distribution and more realistic ecosystem behavior

### 📊 Advanced Generation Tracking
Complete evolutionary lineage analysis:
- **Agent Lineage**: Every agent tracks its generation number (1, 2, 3, etc.)
- **Inheritance Logic**: Offspring inherit `parent.generation + 1`
- **Crossover Tracking**: Offspring inherit `max(parent1.generation, parent2.generation) + 1`
- **Web Integration**: Generation information displayed in agent inspection modals
- **Evolution Analytics**: Track fitness progression across generations

### 🌐 Professional Web Interface
Enterprise-grade web interface for ecosystem monitoring:
- **Live Visualization**: Real-time agent movement with smooth Canvas rendering
- **Agent Inspector**: Interactive neural network analysis with modal system
- **WebSocket Communication**: Instant bidirectional communication for real-time updates
- **Performance Metrics**: Live fitness, energy, population, and neural diversity tracking
- **Responsive Design**: Mobile-friendly interface with modern CSS animations
- **Error Handling**: Graceful fallbacks and connection status indicators

## 📋 Requirements

### Core Dependencies
- **Python 3.8+** - Modern Python features and type hints
- **NumPy ≥1.21.0** - Neural network calculations and matrix operations
- **Matplotlib ≥3.5.0** - Data visualization and chart generation

### Web Interface Dependencies
- **Flask ≥2.0.0** - Web framework for serving the interface
- **Flask-SocketIO ≥5.0.0** - Real-time WebSocket communication
- **Modern Web Browser** - Chrome, Firefox, Safari, or Edge for D3.js support

See `requirements.txt` for complete dependency list with version specifications.

### 🧪 Testing

```bash
# Run comprehensive test suite
python -m pytest tests/

# Test specific neural network components
python tests/test_generation_tracking.py    # Generation inheritance testing
python tests/test_boundary_fix.py           # Boundary awareness validation
python tests/test_ecosystem.py              # Core ecosystem mechanics
python tests/test_carnivore_energy.py       # Energy management testing
python tests/test_comprehensive_carnivore.py # Complete agent behavior validation
```

## 📚 Documentation

### 📖 Comprehensive Documentation
Detailed documentation available in `docs/`:
- **PHASE1_COMPLETE.md**: Basic ecosystem implementation and core mechanics
- **PHASE2_COMPLETE.md**: Neural network integration and decision-making systems
- **PHASE3_COMPLETE.md**: Advanced evolution, web interface, and neural inspection
- **PROJECT_STATUS.md**: Current implementation status and feature roadmap

### � Technical Documentation
- **CHANGELOG.md**: Complete version history with detailed feature descriptions
- **CONTRIBUTING.md**: Development guidelines and contribution instructions
- **Code Comments**: Extensive inline documentation for all neural network components

## �🔄 Development Progress

### ✅ Completed Features (v2.1.0)
- [x] **Real-time Neural Inspection**: Click-to-analyze agent neural networks
- [x] **Advanced Web Interface**: Professional Flask + SocketIO web server
- [x] **D3.js Neural Visualization**: Interactive neural network diagrams
- [x] **Live Data Streaming**: Real-time agent updates via WebSocket
- [x] **Modern Color Scheme**: Professional purple/teal and blue/orange palette
- [x] **Hidden Layer Visualization**: Real-time hidden neuron activation display
- [x] **Generation Tracking**: Complete evolutionary lineage analysis
- [x] **Boundary Awareness**: 10-input neural networks with environmental sensing
- [x] **Comprehensive Testing**: Full test suite with neural network validation
- [x] **Performance Optimization**: Efficient real-time updates and rendering

### 🚀 Future Enhancements (v2.2.0+)
- [ ] **Neural Network Export**: Save/load evolved neural network architectures
- [ ] **Multi-species Evolution**: Different agent types with specialized behaviors
- [ ] **Environmental Challenges**: Dynamic weather, food scarcity, and seasonal changes
- [ ] **Advanced Genetic Operators**: Speciation, elitism, and adaptive mutation rates
- [ ] **Machine Learning Integration**: TensorFlow/PyTorch backend options
- [ ] **GPU Acceleration**: CUDA support for large-scale simulations
- [ ] **Distributed Computing**: Multi-environment connected ecosystems

### 🔬 Research Opportunities
- [ ] **Evolutionary Tree Visualization**: Phylogenetic tree generation from lineage data
- [ ] **Behavioral Analysis**: Pattern recognition in evolved behaviors
- [ ] **Neural Architecture Search**: Automatic neural network architecture optimization
- [ ] **Emergent Communication**: Inter-agent communication protocol evolution

## 🤝 Contributing

We welcome contributions to enhance the neural ecosystem simulation! 

### 🚀 Getting Started
1. **Fork the repository** on GitHub
2. **Clone your fork**: `git clone https://github.com/yourusername/EA-NN.git`
3. **Create a feature branch**: `git checkout -b feature/amazing-enhancement`
4. **Install dependencies**: `pip install -r requirements.txt`
5. **Run tests**: `python -m pytest tests/` to ensure everything works
6. **Make your changes** with comprehensive testing
7. **Commit your changes**: `git commit -m 'Add amazing enhancement'`
8. **Push to your branch**: `git push origin feature/amazing-enhancement`
9. **Open a Pull Request** with detailed description of your changes

### 🎯 Contribution Areas
- **Neural Network Enhancements**: Improve decision-making algorithms
- **Visualization Features**: Add new real-time analysis tools
- **Performance Optimization**: Enhance simulation speed and efficiency
- **Testing Coverage**: Expand test suite for better reliability
- **Documentation**: Improve code documentation and user guides
- **Bug Fixes**: Address issues and improve stability

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete details.

## 🎉 Acknowledgments

Built with passion for **artificial intelligence**, **evolutionary computation**, and **ecosystem modeling**. This simulation demonstrates the emergence of intelligent behavior through neural networks and genetic algorithms in a competitive environment.

### 🏆 Technical Achievements
- **Advanced Neural Architecture**: 10→12→4 feedforward networks with boundary awareness
- **Real-time Visualization**: Professional web interface with D3.js neural diagrams
- **Evolutionary Intelligence**: Sophisticated genetic algorithms with fitness-based selection
- **Interactive Analysis**: Click-to-inspect neural network functionality
- **Modern Design**: Professional color schemes and responsive user interface

### 🌟 Educational Value
Perfect for researchers, students, and AI enthusiasts interested in:
- **Neural Network Behavior**: Understanding how neural networks make decisions
- **Evolutionary Algorithms**: Seeing genetic algorithms in action
- **Emergent Intelligence**: Observing complex behaviors from simple rules
- **Real-time Systems**: Learning WebSocket and modern web technologies
- **Data Visualization**: Exploring D3.js and interactive data display

---

🧠 **Start exploring neural evolution today!** Run `python main.py --web` and watch intelligence emerge!
