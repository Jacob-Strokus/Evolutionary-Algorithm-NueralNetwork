# Neural Ecosystem Simulation

An ecosystem simulation featuring neural network agents that learn food-seeking behavior through genetic algorithms and evolutionary pressure.

## Features

- Unit vector directional inputs for precise food location
- Movement alignment fitness system with real-time behavior analysis
- Mathematical safety systems (infinite/NaN protection)
- Genetic evolution with elite preservation and tournament selection
- Real-time web interface with neural network inspection
- Performance tracking and evolution analytics

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd EA-NN

# Install dependencies
pip install -r requirements.txt

# Run the enhanced evolution system
python main_evolution.py

# Run the web interface
python main.py --web
# Then open: http://localhost:5000
```

## Repository Structure

```
EA-NN/
├── src/                    # Core simulation engine
│   ├── core/              # Ecosystem mechanics and base classes
│   ├── neural/            # Neural network system
│   ├── evolution/         # Genetic algorithms
│   ├── visualization/     # Web interfaces
│   └── analysis/          # Evolution tracking tools
├── demos/                 # Interactive demonstrations
├── examples/              # Ready-to-run examples
├── tests/                 # Testing suite
├── scripts/               # Utility scripts
├── docs/                  # Documentation
├── main.py               # Interactive web interface
├── main_evolution.py     # Evolution system
└── README.md             # This file
```

## Neural Network Architecture

### Network Structure
- **10→12→4 Architecture**: Feedforward neural network for ecosystem survival
- **10 Sensory Inputs**:
  - Energy level (0.0-1.0)
  - Age factor (0.0-1.0)
  - Food distance (0.0-1.0)
  - Food direction X/Y (unit vectors)
  - Threat/prey distance (0.0-1.0)
  - Threat/prey direction X/Y (unit vectors)
  - Boundary safety X/Y (0.0-1.0)
- **12-Neuron Hidden Layer**: Complex decision processing with tanh activation
- **4 Output Actions**: Movement X/Y, reproduction decision, movement intensity

### Fitness System
- Food proximity rewards: +20 points for being near food
- Food consumption bonus: +50 points for successful acquisition
- Movement alignment: +25 points for moving toward food, -10 penalty for moving away
- Energy management: Fitness based on energy-to-maximum ratio
- Species bonuses: Herbivores +15 for efficiency, Carnivores +25 for hunting
- Reproductive success: +30 points per offspring

### Genetic Evolution
- Elite preservation: Top 20% survive automatically
- Tournament selection: Size-4 tournaments for selection pressure
- Crossover rate: 80% genetic recombination
- Mutation rate: 15% with 0.25 strength
- Diversity bonus: 15% for maintaining genetic diversity
- Generation length: 400 steps for evaluation

## Usage Examples

### Enhanced Evolution System
```bash
python main_evolution.py
# Runs optimized evolution with enhanced neural networks
# - 10 generations with 300 steps each
# - Real-time fitness progression tracking
# - Elite preservation with tournament selection
```

### Enhanced Neural Demo
```bash
python demos/enhanced_neural_food_seeking_demo.py
# Interactive demonstration showing:
# - Individual agent neural analysis
# - Movement alignment calculations
# - Multi-generation evolution tracking
```

### Web Interface
```bash
python main.py
# Then open: http://localhost:5000
# - Real-time ecosystem visualization
# - Click agents for neural network inspection
# - Live fitness and population tracking
```

### Neural Learning Examples
```bash
python examples/main_neural.py
# Various neural network demonstrations:
# 1. Basic Neural Network Simulation
# 2. Traditional vs Neural Comparison
# 3. Neural Learning Demo with fitness tracking
# 4. Extended Neural Simulation
```

## Requirements

### Core Dependencies
- Python 3.8+
- NumPy ≥1.21.0
- Matplotlib ≥3.5.0

### Web Interface
- Flask ≥2.0.0
- Flask-SocketIO ≥5.0.0
- Modern web browser for D3.js support

See `requirements.txt` for complete dependency list.

## Testing

```bash
# Run comprehensive test suite
python -m pytest tests/

# Test specific components
python tests/test_ecosystem.py              # Core ecosystem mechanics
python tests/test_carnivore_energy.py       # Energy management
python tests/test_genetic_tracking.py       # Evolution tracking
```

## Documentation

Detailed documentation available in `docs/`:
- **PHASE1_COMPLETE.md**: Basic ecosystem implementation
- **PHASE2_COMPLETE.md**: Neural network integration
- **PHASE3_COMPLETE.md**: Advanced features and web interface
- **PROJECT_STATUS.md**: Current implementation status
- **CHANGELOG.md**: Version history

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/enhancement`
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `python -m pytest tests/`
5. Make your changes with comprehensive testing
6. Commit changes: `git commit -m 'Add enhancement'`
7. Push to branch: `git push origin feature/enhancement`
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
