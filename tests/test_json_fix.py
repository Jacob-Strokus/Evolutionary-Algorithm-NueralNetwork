#!/usr/bin/env python3
"""
GPU JSON Serialization Fix Test
===============================

Test if the JSON serialization fixes work correctly for GPU acceleration.
"""

import os
import sys
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_json_fix():
    """Test JSON serialization with GPU environment"""
    print("🧪 JSON Serialization Fix Test")
    print("=" * 35)
    
    try:
        # Test web server JSON helper
        from src.visualization.web_server import json_serializable
        import numpy as np
        
        print("✅ Testing JSON helper function...")
        
        # Test with numpy types
        test_data = {
            'float32': np.float32(3.14),
            'float64': np.float64(2.71),
            'int32': np.int32(42),
            'int64': np.int64(100),
            'array': np.array([1.0, 2.0, 3.0]),
            'nested': {
                'values': [np.float32(1.1), np.float32(2.2)],
                'count': np.int32(5)
            }
        }
        
        # Convert to JSON-serializable
        clean_data = json_serializable(test_data)
        
        # Test JSON serialization
        json_str = json.dumps(clean_data)
        print(f"✅ JSON serialization successful ({len(json_str)} chars)")
        
        # Test main.py helper
        from main import ensure_json_serializable
        
        clean_data2 = ensure_json_serializable(test_data)
        json_str2 = json.dumps(clean_data2)
        print(f"✅ Main.py JSON helper successful ({len(json_str2)} chars)")
        
        print("\n🎉 All JSON serialization fixes working!")
        return True
        
    except Exception as e:
        print(f"❌ JSON fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎮 Testing JSON serialization fixes for RTX 2060 GPU\n")
    
    success = test_json_fix()
    
    if success:
        print("\n🚀 Ready to run: ./run_gpu.sh --web")
        print("Web interface should now work without JSON errors!")
    else:
        print("\n❌ JSON serialization still needs attention")
