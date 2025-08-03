# Scripts Directory Organization

This directory contains various scripts for testing, analysis, and debugging the EA-NN ecosystem simulation. The scripts have been organized into logical subfolders for better maintainability and navigation.

## 📁 Directory Structure

### 🐺 carnivore_analysis/
Scripts for analyzing carnivore behavior, energy mechanics, and population dynamics:
- `analyze_carnivore_reproduction_issue.py` - Problem analysis for carnivore reproduction issues
- `analyze_carnivore_starvation.py` - Analysis of carnivore survival without food
- `detailed_carnivore_analysis.py` - Comprehensive carnivore behavior analysis
- `investigate_carnivore_food.py` - Investigation of carnivore food consumption patterns
- `track_carnivore_energy.py` - Energy tracking and debugging for carnivores
- `track_carnivore_extinction.py` - Analysis of early carnivore extinction patterns

### 🧪 carnivore_testing/
Test scripts for validating carnivore balance fixes and reproduction mechanics:
- `final_balance_test.py` - Final verification of carnivore balance fixes
- `test_adjusted_reproduction.py` - Parameter validation for reproduction requirements
- `test_carnivore_fixes.py` - Comprehensive ecosystem testing for carnivore fixes
- `test_hunting_effectiveness.py` - Testing carnivore hunting mechanics and effectiveness

### 🔧 carnivore_fixes/
Scripts that implement fixes for carnivore-related issues:
- `fix_carnivore_balance.py` - Implementation of carnivore balance fixes

### 🧠 neural_visualization/
Scripts related to neural network visualization debugging and testing:
- `investigate_neural_viz_bug.py` - Investigation script for neural visualization issues
- `neural_viz_bug_fix.py` - Bug analysis and fix recommendations
- `test_neural_viz_fix.py` - Comprehensive test for neural visualization fixes
- `test_real_web_server.py` - Real web server testing
- `test_real_web_server_neural_fix.py` - Web server testing with neural fixes
- `test_web_server_neural_bug.py` - Web server neural bug simulation
- `validate_neural_fix.py` - Quick validation of neural visualization fixes

### 🗺️ boundary_clustering/
Scripts dealing with boundary clustering and spatial behavior issues:
- `boundary_clustering_investigation.py` - Investigation of boundary clustering behavior
- `debug_boundary_clustering.py` - Debugging boundary clustering issues
- `fix_boundary_clustering.py` - Implementation of boundary clustering fixes
- `train_boundary_awareness.py` - Training agents for better boundary awareness
- `verify_boundary_fix.py` - Verification of boundary clustering fixes

### 📊 general_analysis/
General analysis scripts for evolution and behavioral patterns:
- `analyze_evolution_fitness.py` - Analysis of evolutionary fitness patterns
- `behavioral_fixes_analysis.py` - Analysis of behavioral fixes and improvements

### 🔍 debugging/
General debugging scripts for hunting and step-by-step analysis:
- `debug_hunting.py` - Debugging hunting mechanics
- `debug_step.py` - Step-by-step debugging utility
- `detailed_hunting_analysis.py` - Detailed analysis of hunting behavior

### 🛠️ utilities/
Utility scripts for maintenance and cleanup:
- `cleanup_learning_files.py` - Cleanup script for learning-related files

## 🚀 Usage

To run any script, navigate to the EA-NN root directory and use:

```bash
python scripts/subfolder/script_name.py
```

For example:
```bash
python scripts/carnivore_testing/test_carnivore_fixes.py
python scripts/neural_visualization/validate_neural_fix.py
```

## 📈 Development History

This organization reflects the iterative development process of the EA-NN project:

1. **Phase 1**: Basic ecosystem setup and initial debugging
2. **Phase 2**: Carnivore balance issues identification and fixing
3. **Phase 3**: Neural network visualization bug fixes
4. **Phase 4**: Boundary clustering and spatial behavior improvements

Each subfolder contains scripts that were developed to address specific phases of development, making it easier to understand the project's evolution and maintain related functionality together.

## 🔄 Maintenance

When adding new scripts:
1. Determine the primary focus/category of the script
2. Place it in the appropriate subfolder
3. Update this README if creating a new category
4. Ensure the script follows the project's coding standards

This organization helps maintain a clean and navigable codebase while preserving the development history and making it easier for contributors to find relevant scripts for specific functionality areas.
