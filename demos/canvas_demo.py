"""
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
    
    choice = input("\nSelect visualization (1-3): ").strip()
    
    if choice == "1":
        print("\n🖼️ Starting simple canvas...")
        run_realtime_canvas("simple")
    elif choice == "2":
        print("\n📊 Starting advanced canvas...")
        run_realtime_canvas("advanced")
    else:
        print("\n👋 Exiting demo")

if __name__ == "__main__":
    main()
