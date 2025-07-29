#!/usr/bin/env python3
"""
Simple test to verify genetic operations tracking is working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment
import time

def test_genetic_counter_bug():
    """Test that total counters accumulate correctly"""
    print("🧬 Testing Genetic Counter Accumulation")
    print("=" * 50)
    
    # Create environment
    env = NeuralEnvironment(width=200, height=200, use_neural_agents=True)
    genetic_alg = env.genetic_algorithm
    
    print("Initial state:")
    stats = genetic_alg.get_genetic_stats()
    print(f"  Total mutations: {stats['total_mutations']}")
    print(f"  Total crossovers: {stats['total_crossovers']}")
    print(f"  Recent mutations: {stats['recent_mutations']}")
    print(f"  Recent crossovers: {stats['recent_crossovers']}")
    
    # Run simulation to generate some genetic events
    print("\n🔄 Running 10 simulation steps...")
    for step in range(10):
        env.step()
        
        # Check stats every few steps
        if step % 5 == 4:  # Steps 4, 9
            stats = genetic_alg.get_genetic_stats()
            print(f"\nAfter step {step + 1}:")
            print(f"  Total mutations: {stats['total_mutations']}")
            print(f"  Total crossovers: {stats['total_crossovers']}")
            print(f"  Recent mutations: {stats['recent_mutations']}")
            print(f"  Recent crossovers: {stats['recent_crossovers']}")
            print(f"  Mutations list length: {len(genetic_alg.genetic_events['mutations'])}")
            print(f"  Crossovers list length: {len(genetic_alg.genetic_events['crossovers'])}")
            
            # Reset recent counters (like the canvas does)
            print("  🔄 Resetting recent counters...")
            genetic_alg.reset_recent_counters()
            
            # Check totals after reset
            stats_after = genetic_alg.get_genetic_stats()
            print(f"  After reset - Total mutations: {stats_after['total_mutations']}")
            print(f"  After reset - Total crossovers: {stats_after['total_crossovers']}")
            print(f"  After reset - Recent mutations: {stats_after['recent_mutations']}")
            print(f"  After reset - Recent crossovers: {stats_after['recent_crossovers']}")
    
    print("\n✅ Test completed!")
    print("\n📊 Final Analysis:")
    final_stats = genetic_alg.get_genetic_stats()
    print(f"  Final total mutations: {final_stats['total_mutations']}")
    print(f"  Final total crossovers: {final_stats['total_crossovers']}")
    print(f"  Events list lengths: {len(genetic_alg.genetic_events['mutations'])}, {len(genetic_alg.genetic_events['crossovers'])}")
    
    # The totals should be >= the list lengths (because totals never decrease)
    mutations_consistent = final_stats['total_mutations'] >= len(genetic_alg.genetic_events['mutations'])
    crossovers_consistent = final_stats['total_crossovers'] >= len(genetic_alg.genetic_events['crossovers'])
    
    print(f"\n✅ Mutations total consistency: {mutations_consistent}")
    print(f"✅ Crossovers total consistency: {crossovers_consistent}")
    
    if mutations_consistent and crossovers_consistent:
        print("🎉 Bug fixed! Total counters are accumulating correctly.")
    else:
        print("❌ Bug still exists! Total counters are not accumulating properly.")
    
    return mutations_consistent and crossovers_consistent

if __name__ == "__main__":
    try:
        success = test_genetic_counter_bug()
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Tests failed!")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
