"""
Pure Evolution Cleanup Script
Removes obsolete learning-related files since pure evolution outperforms learning
"""
import os
import sys

def cleanup_obsolete_learning_files():
    """Remove files that are no longer needed with pure evolution system"""
    
    print("🧹 PURE EVOLUTION CLEANUP")
    print("=" * 50)
    print("Removing obsolete learning files since evolution outperforms learning")
    print()
    
    # Files to remove (learning-related files that are no longer used)
    files_to_remove = [
        # Learning system files
        "src/neural/online_learning.py",           # Original online learning system
        "src/neural/improved_online_learning.py", # Improved learning (still underperformed)
        "src/neural/learning_agents.py",          # Learning agent wrappers
        
        # Learning demo files  
        "demos/enhanced_learning_demo.py",        # Original learning comparison demo
        "demos/improved_learning_demo.py",        # Improved learning demo
        
        # Analysis results from learning experiments
        "learning_comparison_results.png",        # Chart showing learning underperformed
        "improved_learning_comparison.png",       # Chart showing improved learning still underperformed
    ]
    
    # Files to keep (analysis and research files)
    files_to_keep = [
        "scripts/analyze_evolution_fitness.py",   # Evolution analysis tool
        "demos/pure_evolution_demo.py",          # Pure evolution demo
        "main_evolution.py",                      # Enhanced pure evolution system
        "pure_evolution_results.png",            # Evolution results chart
    ]
    
    print("📁 FILES TO REMOVE:")
    removed_count = 0
    kept_for_reference = []
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            print(f"  ❌ {file_path}")
            # Don't actually remove yet, just simulate
            # os.remove(file_path)  # Uncomment to actually remove
            removed_count += 1
        else:
            print(f"  ⚠️  {file_path} (not found)")
    
    print(f"\n📁 FILES TO KEEP:")
    for file_path in files_to_keep:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️  {file_path} (not found)")
    
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"  Files marked for removal: {len(files_to_remove)}")
    print(f"  Files that exist and would be removed: {removed_count}")
    print(f"  Files to keep: {len(files_to_keep)}")
    
    return files_to_remove, files_to_keep

def analyze_dependencies():
    """Analyze which files still have dependencies on learning modules"""
    
    print("\n🔍 DEPENDENCY ANALYSIS")
    print("=" * 30)
    
    # Check for remaining imports of learning modules
    learning_imports = [
        "from src.neural.online_learning",
        "from src.neural.learning_agents", 
        "from src.neural.improved_online_learning",
        "import.*learning",
        "LearningAgent",
        "LearningConfig",
        "OnlineLearning"
    ]
    
    # Files to check (core system files)
    core_files = [
        "main.py",
        "main_evolution.py", 
        "src/neural/neural_agents.py",
        "src/neural/neural_network.py",
        "src/core/ecosystem.py",
        "src/evolution/genetic_evolution.py",
        "src/evolution/advanced_genetic.py",
        "src/visualization/monitor.py",
        "src/visualization/neural_visualizer.py"
    ]
    
    dependencies_found = False
    
    for file_path in core_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                
            has_learning_imports = False
            for import_pattern in learning_imports:
                if import_pattern in content:
                    if not has_learning_imports:
                        print(f"\n📄 {file_path}:")
                        has_learning_imports = True
                        dependencies_found = True
                    print(f"  ⚠️  Found: {import_pattern}")
    
    if not dependencies_found:
        print("\n✅ No learning dependencies found in core files!")
        print("   Core system is clean and ready for pure evolution.")
    
    return not dependencies_found

def create_clean_project_structure():
    """Show the clean project structure after removing learning files"""
    
    print("\n📁 CLEAN PROJECT STRUCTURE")
    print("=" * 40)
    
    clean_structure = """
    EA-NN/
    ├── main.py                          # Original main entry
    ├── main_evolution.py               # ✨ Enhanced pure evolution system
    ├── requirements.txt
    ├── README.md
    ├── CHANGELOG.md
    │
    ├── src/
    │   ├── core/
    │   │   └── ecosystem.py            # Core ecosystem simulation
    │   ├── neural/
    │   │   ├── neural_network.py       # Neural network implementation
    │   │   └── neural_agents.py        # Neural agents with evolution fitness
    │   ├── evolution/
    │   │   ├── genetic_evolution.py    # Core genetic algorithms
    │   │   └── advanced_genetic.py     # Advanced genetic operations
    │   └── visualization/
    │       ├── monitor.py              # Performance monitoring
    │       ├── neural_visualizer.py    # Neural network visualization
    │       └── web_server.py           # Web interface
    │
    ├── demos/
    │   ├── pure_evolution_demo.py      # ✨ Pure evolution demonstration
    │   └── canvas_demo.py              # Visualization demo
    │
    ├── scripts/
    │   ├── analyze_evolution_fitness.py # ✨ Evolution fitness analysis
    │   └── [other analysis scripts]
    │
    └── tests/
        └── [test files]
    
    ✨ = New/Enhanced for pure evolution
    ❌ = Removed (learning files)
    """
    
    print(clean_structure)

def main():
    """Run the cleanup analysis"""
    
    files_to_remove, files_to_keep = cleanup_obsolete_learning_files()
    is_clean = analyze_dependencies()
    create_clean_project_structure()
    
    print("\n🎯 RECOMMENDATIONS:")
    print("-" * 20)
    
    if is_clean:
        print("✅ Core system is clean - no learning dependencies")
        print("✅ Safe to remove learning files")
        print("✅ Pure evolution system is standalone")
    else:
        print("⚠️  Found learning dependencies in core files")
        print("⚠️  Review dependencies before removing files")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Review the files marked for removal")
    print("2. Ensure no critical functionality will be lost") 
    print("3. Update imports in any remaining files if needed")
    print("4. Run tests to verify system works without learning files")
    print("5. Create feature branch and commit changes")
    
    print(f"\n💡 To actually remove files, uncomment the os.remove() line in this script")

if __name__ == "__main__":
    main()
