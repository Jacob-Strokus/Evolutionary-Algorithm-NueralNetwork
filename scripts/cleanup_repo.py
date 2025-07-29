#!/usr/bin/env python3
"""
Repository Cleanup and Organization Script
Cleans up temporary files, organizes codebase, and creates proper structure
"""

import os
import shutil
from pathlib import Path

def cleanup_repository():
    """Clean and organize the entire repository"""
    
    print("🧹 Starting Repository Cleanup")
    print("=" * 50)
    
    # Files to remove (temporary, debugging, obsolete)
    files_to_remove = [
        # Temporary visualization files
        "advanced_ecosystem_display.html",
        "realtime_advanced_ecosystem.png",
        "neural_ecosystem_step_200.png",
        
        # Debugging scripts (no longer needed)
        "diagnostic.py",
        "quick_test.py",
        
        # Obsolete runner scripts
        "run_enhanced_simulation.py",
        "run_clean_simulation.py",
        
        # Shell scripts (WSL specific, not needed for main repo)
        "setup_gui.sh",
        
        # Auto-generated step images
        *[f"advanced_ecosystem_step_{i:03d}.png" for i in range(0, 100, 10)],
    ]
    
    # Directories to clean
    dirs_to_clean = [
        "__pycache__",
        "examples/__pycache__",
        "src/__pycache__",
        "src/visualization/__pycache__",
        "src/core/__pycache__",
        "src/neural/__pycache__",
        "src/evolution/__pycache__",
        "src/analysis/__pycache__",
    ]
    
    # Remove temporary files
    print("\n🗑️ Removing temporary and debugging files...")
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"   ✅ Removed: {file_path}")
    
    # Clean __pycache__ directories
    print("\n🧹 Cleaning Python cache directories...")
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"   ✅ Cleaned: {dir_path}")
    
    # Create organized examples directory
    print("\n📁 Organizing examples...")
    organize_examples()
    
    # Update main runner script
    print("\n🔧 Creating clean main runner...")
    create_main_runner()
    
    # Create organized visualization demos
    print("\n🎨 Creating visualization demos...")
    create_visualization_demos()
    
    # Update documentation
    print("\n📝 Updating documentation...")
    update_documentation()
    
    print("\n✅ Repository cleanup complete!")
    print("\nNew structure:")
    print("  📁 src/           - Core source code")
    print("  📁 examples/      - Usage examples")
    print("  📁 demos/         - Visualization demos")
    print("  📁 docs/          - Documentation")
    print("  📁 tests/         - Unit tests")
    print("  🐍 main.py        - Primary entry point")
    print("  🐍 requirements.txt - Dependencies")

def organize_examples():
    """Organize the examples directory"""
    
    # Keep only essential examples
    essential_examples = {
        "main_neural.py": "Neural ecosystem with learning agents",
        "main.py": "Basic ecosystem simulation"
    }
    
    # Remove obsolete example files
    examples_dir = Path("examples")
    if examples_dir.exists():
        for file in examples_dir.iterdir():
            if file.name not in essential_examples and file.suffix == ".py":
                file.unlink()
                print(f"   ✅ Removed obsolete example: {file.name}")

def create_main_runner():
    """Create a clean main entry point"""
    
    main_content = '''"""
AI Neural Ecosystem Simulation
Main entry point for the ecosystem simulation
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.neural.neural_agents import NeuralEnvironment
from src.visualization.realtime_canvas import run_realtime_canvas

def main():
    """Main application entry point"""
    
    print("🧬 AI Neural Ecosystem Simulation")
    print("=" * 40)
    print("1. Quick Demo (50 steps)")
    print("2. Simple Real-time Canvas")
    print("3. Advanced Real-time Canvas")
    print("4. Headless Simulation")
    print("5. Exit")
    
    choice = input("\\nSelect option (1-5): ").strip()
    
    if choice == "1":
        print("\\n🚀 Running quick demo...")
        quick_demo()
    elif choice == "2":
        print("\\n🎨 Starting simple canvas...")
        run_realtime_canvas("simple")
    elif choice == "3":
        print("\\n🎨 Starting advanced canvas...")
        run_realtime_canvas("advanced")
    elif choice == "4":
        print("\\n🤖 Running headless simulation...")
        headless_simulation()
    elif choice == "5":
        print("\\n👋 Goodbye!")
        return
    else:
        print("\\n❌ Invalid choice, running demo...")
        quick_demo()

def quick_demo():
    """Run a quick 50-step demonstration"""
    
    env = NeuralEnvironment(width=150, height=150, use_neural_agents=True)
    
    print("\\nRunning 50-step neural ecosystem demo...")
    print("Watch as agents learn and adapt!")
    print("-" * 40)
    
    for step in range(50):
        env.step()
        
        if step % 10 == 0:
            stats = env.get_neural_stats()
            print(f"Step {step:2d}: "
                  f"H={stats['herbivore_count']:2d}, "
                  f"C={stats['carnivore_count']:2d}, "
                  f"Fitness={stats.get('avg_neural_fitness', 0):.1f}")
    
    print("\\n✅ Demo complete!")
    final_stats = env.get_neural_stats()
    print(f"Final fitness: {final_stats.get('avg_neural_fitness', 0):.1f}")

def headless_simulation():
    """Run a longer headless simulation with detailed output"""
    
    env = NeuralEnvironment(width=200, height=200, use_neural_agents=True)
    
    print("\\nRunning 200-step headless simulation...")
    print("Detailed learning progress:")
    print("-" * 50)
    
    fitness_history = []
    
    for step in range(200):
        env.step()
        
        if step % 20 == 0:
            stats = env.get_neural_stats()
            fitness = stats.get('avg_neural_fitness', 0)
            fitness_history.append(fitness)
            
            print(f"Step {step:3d}: "
                  f"H={stats['herbivore_count']:2d}, "
                  f"C={stats['carnivore_count']:2d}, "
                  f"Fitness={fitness:.1f}, "
                  f"Decisions={stats.get('avg_decisions_made', 0):.0f}")
    
    print("\\n📊 Learning Analysis:")
    if len(fitness_history) > 1:
        improvement = fitness_history[-1] - fitness_history[0]
        print(f"   Initial fitness: {fitness_history[0]:.1f}")
        print(f"   Final fitness: {fitness_history[-1]:.1f}")
        print(f"   Total improvement: {improvement:+.1f}")
        
        if improvement > 5:
            print("   🎉 Excellent learning progress!")
        elif improvement > 1:
            print("   ✅ Good learning progress!")
        else:
            print("   📈 Steady learning progress!")

if __name__ == "__main__":
    main()
'''
    
    with open("main.py", "w") as f:
        f.write(main_content)
    print("   ✅ Created: main.py")

def create_visualization_demos():
    """Create organized visualization demos"""
    
    # Create demos directory
    demos_dir = Path("demos")
    demos_dir.mkdir(exist_ok=True)
    
    # Create canvas launcher demo
    canvas_demo = '''"""
Visualization Canvas Demo
Demonstrates different visualization modes
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.visualization.realtime_canvas import run_realtime_canvas

def main():
    """Canvas demo launcher"""
    
    print("🎨 Visualization Canvas Demos")
    print("=" * 35)
    print("1. Simple Real-time Canvas")
    print("2. Advanced Multi-panel Canvas")
    print("3. Exit")
    
    choice = input("\\nSelect visualization (1-3): ").strip()
    
    if choice == "1":
        print("\\n🖼️ Starting simple canvas...")
        run_realtime_canvas("simple")
    elif choice == "2":
        print("\\n📊 Starting advanced canvas...")
        run_realtime_canvas("advanced")
    else:
        print("\\n👋 Exiting demo")

if __name__ == "__main__":
    main()
'''
    
    with open("demos/canvas_demo.py", "w") as f:
        f.write(canvas_demo)
    print("   ✅ Created: demos/canvas_demo.py")

def update_documentation():
    """Update the main README with clean structure"""
    
    readme_content = '''# AI Neural Ecosystem Simulation

An advanced ecosystem simulation featuring neural network-powered agents that learn and evolve in real-time.

## Features

🧠 **Neural Learning**: Agents use feedforward neural networks to make decisions
🧬 **Evolution**: Genetic algorithms improve agent performance over generations  
🎮 **Real-time Visualization**: Interactive displays show learning progress
📊 **Multi-panel Analytics**: Advanced charts track fitness, population, and neural activity

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run main simulation
python main.py
```

## Project Structure

```
src/
├── core/           # Core ecosystem mechanics
├── neural/         # Neural network agents
├── evolution/      # Genetic algorithms
├── visualization/  # Real-time displays
└── analysis/       # Data analysis tools

examples/           # Usage examples
demos/             # Visualization demos
docs/              # Documentation
tests/             # Unit tests
```

## Usage Examples

### Quick Demo
```python
python main.py  # Choose option 1
```

### Real-time Visualization
```python
python main.py  # Choose option 2 or 3
```

### Canvas Demos
```python
python demos/canvas_demo.py
```

## Learning Progress

Agents start with random behaviors and learn through:
- Neural network decision making
- Fitness-based selection
- Real-time adaptation
- Environmental feedback

Watch fitness scores improve as agents become smarter!

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- See `requirements.txt` for full list

## Documentation

See `docs/` directory for detailed documentation:
- Phase completion reports
- API documentation  
- Architecture overview
'''
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("   ✅ Updated: README.md")

if __name__ == "__main__":
    cleanup_repository()
