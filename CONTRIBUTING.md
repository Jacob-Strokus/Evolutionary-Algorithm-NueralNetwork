# Contributing to Neural Ecosystem Simulation

Thank you for your interest in contributing! This project welcomes contributions from the community.

## 🚀 Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/coolshit.git
   cd coolshit
   ```
3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 🧪 Testing

Before submitting any changes, make sure all tests pass:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python tests/test_ecosystem.py          # Core ecosystem tests
python tests/test_generation_tracking.py  # Generation tracking tests
python tests/test_boundary_fix.py       # Boundary awareness tests
```

## 📝 Code Style

- Follow PEP 8 Python style guidelines
- Use descriptive variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular
- Comment complex algorithms and neural network logic

## 🧬 Areas for Contribution

### 🎯 High Priority
- **Performance Optimization**: Improve simulation speed for larger populations
- **Advanced Visualizations**: New charts and analytics for evolutionary progress
- **Neural Network Improvements**: Additional sensory inputs or network architectures
- **Testing**: Expand test coverage for edge cases

### 🌟 Feature Ideas
- **Multi-Environment**: Multiple connected ecosystems
- **Environmental Challenges**: Seasonal changes, disasters, resource scarcity
- **Advanced Genetics**: Sexual reproduction, genetic diversity metrics
- **Export/Import**: Save and load evolved populations
- **Machine Learning Integration**: TensorFlow/PyTorch backend options

### 🐛 Bug Reports
When reporting bugs, please include:
- Python version and operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Console output or error messages
- Screenshots (for visualization issues)

## 🔧 Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   python -m pytest tests/
   python examples/main_neural.py  # Quick functionality test
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📚 Code Structure

### Core Components
- **`src/core/`**: Base ecosystem mechanics (agents, environment, food)
- **`src/neural/`**: Neural network implementations and agent intelligence
- **`src/evolution/`**: Genetic algorithms and evolutionary operations
- **`src/visualization/`**: Real-time displays and web interface
- **`src/analysis/`**: Data analysis and performance monitoring

### Key Files
- **`neural_agents.py`**: Main neural agent implementation
- **`neural_network.py`**: Neural network architecture and sensory systems
- **`genetic_evolution.py`**: Genetic algorithms and crossover operations
- **`advanced_web_server.py`**: Advanced web interface with neural network inspection

## 🎨 Adding New Features

### Neural Network Enhancements
When adding new sensory inputs or neural capabilities:
1. Update the `SensorSystem.get_sensory_inputs()` method
2. Modify neural network input layer size accordingly
3. Add comprehensive tests for new inputs
4. Update documentation with new sensory descriptions

### Visualization Features
For new visualization components:
1. Follow the existing web server architecture
2. Add both JavaScript frontend and Python backend support
3. Ensure real-time updates work correctly
4. Test across different browsers

### Genetic Algorithm Improvements
When enhancing evolution:
1. Maintain backward compatibility with existing fitness functions
2. Add configurable parameters for new genetic operations
3. Test with various population sizes and evolution parameters
4. Document new evolutionary strategies

## 🤝 Community

- Be respectful and inclusive in all interactions
- Help newcomers get started with the codebase
- Share interesting results or evolved behaviors
- Suggest improvements for documentation and usability

## 📋 Pull Request Guidelines

- **Clear Title**: Describe what your PR does in one line
- **Detailed Description**: Explain the changes and why they're needed
- **Screenshots**: Include visuals for UI/visualization changes
- **Testing**: Confirm all tests pass and add new tests if needed
- **Documentation**: Update relevant documentation

Example PR title formats:
- `feat: add seasonal environmental changes`
- `fix: resolve boundary clustering in small environments`
- `docs: update neural network architecture documentation`
- `test: add comprehensive genetic algorithm test suite`

Thank you for contributing to the Neural Ecosystem Simulation! 🧠🌟
