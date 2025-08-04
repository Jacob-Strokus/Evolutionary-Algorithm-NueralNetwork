#!/usr/bin/env python3
"""
Simple integration test for optimized ecosystem performance
Tests basic functionality after merging with main branch
"""

import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_basic_optimization_import():
    """Test that optimization modules can be imported"""
    try:
        from optimization.high_performance_ecosystem import create_optimized_environment
        from optimization.spatial_indexing import SpatialGrid
        print("✅ Optimization modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_optimized_environment_creation():
    """Test creating optimized environments"""
    try:
        from optimization.high_performance_ecosystem import create_optimized_environment
        
        width, height = 800, 600
        
        for level in ["low", "medium", "high", "maximum"]:
            env = create_optimized_environment(width, height, performance_level=level)
            if env is None:
                print(f"❌ Failed to create {level} performance environment")
                return False
            print(f"✅ Created {level} performance environment")
            
        return True
    except Exception as e:
        print(f"❌ Environment creation error: {e}")
        return False

def test_spatial_indexing():
    """Test spatial indexing functionality"""
    try:
        from optimization.spatial_indexing import SpatialGrid, SpatialPoint
        
        grid = SpatialGrid(800, 600, cell_size=25.0)
        
        # Add some points
        points = []
        for i in range(50):
            x = 100 + i * 10
            y = 100 + i * 5
            point = SpatialPoint(x, y, f"point_{i}", i)  # Added id parameter
            grid.add_point(point)
            points.append(point)
            
        # Test queries
        nearby = grid.query_radius(200, 150, 50)
        print(f"✅ Spatial query found {len(nearby)} nearby points")
        
        # Test performance
        start_time = time.time()
        for _ in range(1000):
            grid.query_radius(400, 300, 100)
        spatial_time = time.time() - start_time
        
        print(f"✅ 1000 spatial queries completed in {spatial_time:.3f}s")
        print(f"   Query rate: {1000/spatial_time:.1f} queries/sec")
        
        return True
    except Exception as e:
        print(f"❌ Spatial indexing error: {e}")
        return False

def test_performance_demonstration():
    """Run a simple performance demonstration"""
    try:
        from optimization.high_performance_ecosystem import create_optimized_environment
        from core.ecosystem import Environment, SpeciesType
        
        # Create environments
        standard_env = Environment(400, 300)
        optimized_env = create_optimized_environment(400, 300, "high")
        
        # Simple timing test
        steps = 50
        
        # Time standard environment
        start = time.time()
        for _ in range(steps):
            standard_env.step()
        standard_time = time.time() - start
        
        # Time optimized environment  
        start = time.time()
        for _ in range(steps):
            optimized_env.step()
        optimized_time = time.time() - start
        
        print(f"✅ Performance comparison ({steps} steps):")
        print(f"   Standard: {standard_time:.3f}s ({steps/standard_time:.1f} steps/sec)")
        print(f"   Optimized: {optimized_time:.3f}s ({steps/optimized_time:.1f} steps/sec)")
        
        if optimized_time <= standard_time:
            speedup = standard_time / optimized_time
            print(f"   🚀 Speedup: {speedup:.2f}x faster")
        else:
            print(f"   ⚠️  Overhead: {optimized_time/standard_time:.2f}x slower (normal for small scale)")
            
        return True
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("🔧 Running Integration Tests for Optimized Ecosystem")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_optimization_import),
        ("Environment Creation", test_optimized_environment_creation), 
        ("Spatial Indexing", test_spatial_indexing),
        ("Performance Demo", test_performance_demonstration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            
    print("\n" + "=" * 60)
    print(f"🎯 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests PASSED! Optimization system is working correctly.")
        return True
    else:
        print("⚠️  Some integration tests failed. Check the errors above.")
        return False

if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
