# 🧠 AI Neural Ecosystem Simulation

An advanced ecosystem simulation featuring neural network-powered agents that learn and evolve in real-time through genetic algorithms.

## ✨ Features

🧠 **Neural Learning**: Agents use 10-input feedforward neural networks to make intelligent decisions  
🧬 **Genetic Evolution**: Sophisticated genetic algorithms with crossover, mutation, and fitness-based selection  
🎮 **Real-time Web Interface**: Interactive web-based visualization with live agent inspection  
📊 **Generation Tracking**: Full generational lineage tracking for evolutionary analysis  
🌐 **Boundary Awareness**: Neural networks include environmental boundary detection to prevent clustering  
📈 **Advanced Analytics**: Multi-panel charts tracking fitness evolution, population dynamics, and neural activity  
🔍 **Agent Inspector**: Click any agent for detailed neural network analysis and performance metrics  

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
├── visualization/     # Real-time displays and web server
└── analysis/          # Data analysis and neural network inspection tools

examples/              # Ready-to-run simulation examples
├── main_neural.py     # Primary neural simulation interface
└── main.py           # Traditional ecosystem comparison

tests/                 # Comprehensive test suite
scripts/               # Utility scripts and debugging tools
docs/                  # Phase completion reports and documentation
```

## 🎯 Usage Examples

### Neural Ecosystem Simulation
```bash
python examples/main_neural.py
# Choose simulation type:
# 1. Basic Neural Network Simulation (500 steps)
# 2. Traditional vs Neural Comparison
# 3. Neural Learning Demo
# 4. Extended Neural Simulation (1500 steps)
```

### Advanced Web-Based Real-time Visualization
```bash
python main.py --web
# Open browser to: http://localhost:5000
```

The advanced neural ecosystem web interface provides:
- **🖥️ Real-time Display**: Live simulation without page refresh
- **🧠 Neural Inspection**: Click any agent to view its neural network structure
- **📊 Live Metrics**: Real-time population, energy, and generation charts
- **⚡ Speed Controls**: Adjust simulation speed (10ms - 500ms per step)
- **📈 Neural Diversity**: Visualize genetic diversity and evolution
- **🎯 D3.js Diagrams**: Interactive neural network visualizations
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **🔄 WebSocket Communication**: Instant updates via WebSocket technology

Key Features:
- Interactive agent inspection (click on any agent)
- Real-time fitness and population charts
- Neural network structure visualization with weights and biases
- Generation tracking and evolutionary lineage
- Customizable simulation speed controls

### Traditional Ecosystem
```bash
python main.py
# Multi-panel traditional ecosystem simulation
```

## 🧬 Neural Architecture

### Agent Intelligence
- **10-Input Sensory System**: Energy, age, food detection, threat detection, population density, reproduction readiness, boundary awareness (X/Y distances)
- **Hidden Layers**: 12-neuron hidden layer for complex decision processing
- **4-Output Actions**: Movement (X/Y), reproduction attempt, and specialized behaviors
- **Adaptive Learning**: Fitness-based genetic selection with neural network evolution

### Evolutionary Mechanics
- **Generation Tracking**: Full lineage tracking from Generation 1 onwards
- **Genetic Operations**: Crossover between compatible agents, mutation with configurable rates
- **Fitness Selection**: Survival-based selection pressure with energy efficiency metrics
- **Population Dynamics**: Dynamic population management with reproduction cooldowns

## 🌟 Key Innovations

### Boundary Clustering Solution
We identified and solved a critical issue where agents would cluster at environment boundaries:
- **Problem**: Neural networks lacked boundary awareness, causing agents to get "stuck" at edges
- **Solution**: Added boundary distance inputs [8] and [9] to the neural sensory system
- **Result**: Agents now naturally avoid boundaries and explore the full environment

### Generation Tracking System
Complete generational tracking for evolutionary analysis:
- **Agent Lineage**: Every agent knows its generation (1, 2, 3, etc.)
- **Inheritance**: Offspring inherit `parent.generation + 1`
- **Crossover**: Offspring inherit `max(parent1.generation, parent2.generation) + 1`
- **Web Display**: Generation information visible in agent inspection cards

### Real-time Web Interface
Professional-grade web interface for ecosystem monitoring:
- **Live Visualization**: Real-time agent movement and behavior
- **Agent Inspector**: Click any agent for detailed analysis
- **Neural Networks**: Interactive neural network structure display
- **Performance Metrics**: Live fitness, energy, and population tracking

## 📋 Requirements

- Python 3.8+
- NumPy ≥1.21.0
- Matplotlib ≥3.5.0
- Flask ≥2.0.0 (for web interface)
- Flask-SocketIO ≥5.0.0 (for real-time updates)

See `requirements.txt` for complete dependency list.

## 🧪 Testing

```bash
# Run comprehensive test suite
python -m pytest tests/

# Test specific components
python tests/test_generation_tracking.py
python tests/test_boundary_fix.py
python tests/test_ecosystem.py
```

## 📚 Documentation

Detailed documentation available in `docs/`:
- **PHASE1_COMPLETE.md**: Basic ecosystem implementation
- **PHASE2_COMPLETE.md**: Neural network integration  
- **PHASE3_COMPLETE.md**: Advanced evolution and web interface
- **PROJECT_STATUS.md**: Current implementation status

## 🔄 Development Progress

### ✅ Completed Features
- [x] Core ecosystem mechanics with predator-prey dynamics
- [x] Neural network decision-making for all agents
- [x] Genetic algorithms with crossover and mutation
- [x] Real-time web visualization with agent inspection
- [x] Generation tracking and evolutionary lineage
- [x] Boundary awareness neural inputs
- [x] Comprehensive test suite
- [x] Advanced analytics and performance monitoring

### 🚀 Future Enhancements
- [ ] Multi-species neural evolution
- [ ] Environmental challenges and seasonal changes
- [ ] Advanced genetic operators (speciation, elitism)
- [ ] Machine learning model export/import
- [ ] Distributed simulation across multiple environments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎉 Acknowledgments

Built with passion for artificial intelligence, evolutionary computation, and ecosystem modeling. This simulation demonstrates the emergence of intelligent behavior through neural networks and genetic algorithms in a competitive environment.
