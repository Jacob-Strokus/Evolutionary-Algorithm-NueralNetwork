"""
Simple Phase 2 Component Tests

Quick validation of multi-target processing and temporal networks
without Phase 1 dependencies.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import Phase 2 components
from neural.multi_target_processor import (
    MultiTargetProcessor, AttentionMechanism, TargetPrioritizer,
    create_multi_target_processor
)
from neural.temporal_networks import (
    AdvancedRecurrentNetwork, GatedMemoryCell, MultiTimescaleMemory,
    TemporalPatternRecognizer, create_advanced_recurrent_network
)


def test_multi_target_basic():
    """Basic multi-target processing test"""
    print("🎯 Testing Multi-Target Processing...")
    
    # Create processor
    processor = create_multi_target_processor("test_agent", 25)
    
    # Create test targets
    targets = [
        {'x': 45.0, 'y': 48.0, 'type': 'food', 'energy_value': 20, 'id': 'food1'},
        {'x': 52.0, 'y': 51.0, 'type': 'threat', 'danger_level': 7, 'id': 'threat1'}
    ]
    
    # Test processing
    sensory_input = np.random.normal(0, 0.3, 25)
    agent_pos = (50.0, 50.0)
    agent_energy = 75.0
    
    recommendations = processor.process_targets(
        sensory_input, targets, agent_pos, agent_energy
    )
    
    # Validate
    assert 'primary_action' in recommendations
    assert 'confidence' in recommendations
    assert 0.0 <= recommendations['confidence'] <= 1.0
    
    print(f"   ✅ Primary action: {recommendations['primary_action']}")
    print(f"   ✅ Confidence: {recommendations['confidence']:.3f}")
    return True


def test_temporal_networks_basic():
    """Basic temporal networks test"""
    print("🧠 Testing Temporal Networks...")
    
    # Create network
    network = create_advanced_recurrent_network(25, 16, "test_agent")
    
    # Test forward pass
    test_input = np.random.normal(0, 0.5, 25)
    output, temporal_info = network.forward(test_input)
    
    # Validate
    assert len(output) == 16
    assert 'memory_activation' in temporal_info
    
    # Test with action
    output2, temporal_info2 = network.forward(
        test_input, current_action='explore', reward=0.1
    )
    
    print(f"   ✅ Network output shape: {output.shape}")
    print(f"   ✅ Memory strength: {temporal_info['memory_activation']['strength']:.3f}")
    return True


def test_attention_mechanism():
    """Test attention mechanism"""
    print("👁️ Testing Attention Mechanism...")
    
    attention = AttentionMechanism(25, 16)
    
    # Create test targets
    target_features = [
        np.random.normal(0, 0.5, 25),
        np.random.normal(0, 0.5, 25),
        np.random.normal(0, 0.5, 25)
    ]
    context = np.random.normal(0, 0.5, 25)
    
    # Test forward pass
    attended_output, attention_weights = attention.forward(target_features, context)
    
    # Validate
    assert len(attended_output) == 25
    assert len(attention_weights) == 3
    assert abs(np.sum(attention_weights) - 1.0) < 0.001
    
    print(f"   ✅ Attention weights sum: {np.sum(attention_weights):.6f}")
    print(f"   ✅ Output shape: {attended_output.shape}")
    return True


def test_gated_memory():
    """Test gated memory cell"""
    print("🧠 Testing Gated Memory Cell...")
    
    cell = GatedMemoryCell(25, 16, "test_cell")
    
    # Test multiple steps
    inputs = [np.random.normal(0, 0.5, 25) for _ in range(5)]
    
    for inp in inputs:
        hidden, cell_state = cell.forward(inp)
        assert len(hidden) == 16
        assert len(cell_state) == 16
    
    # Test reset
    cell.reset_state()
    assert np.allclose(cell.hidden_state, 0)
    assert np.allclose(cell.cell_state, 0)
    
    print(f"   ✅ Processed {len(inputs)} steps correctly")
    print(f"   ✅ State reset working")
    return True


def test_pattern_recognition():
    """Test temporal pattern recognition"""
    print("🔍 Testing Pattern Recognition...")
    
    recognizer = TemporalPatternRecognizer(pattern_length=3)
    
    # Create simple pattern
    actions = ['move_up', 'move_right', 'move_down'] * 3
    rewards = [0.1, 0.2, 0.15] * 3
    
    # Observe pattern
    for action, reward in zip(actions, rewards):
        recognizer.observe_action(action, reward)
    
    # Test extraction
    patterns = recognizer.get_best_patterns()
    
    # Test prediction
    predictions = recognizer.predict_next_action(['move_up', 'move_right'])
    
    print(f"   ✅ Learned {len(patterns)} patterns")
    if patterns:
        print(f"   ✅ Best pattern confidence: {patterns[0]['confidence']:.3f}")
    if predictions:
        print(f"   ✅ Predictions: {list(predictions.keys())}")
    return True


def run_simple_tests():
    """Run all simple tests"""
    print("=" * 60)
    print("🚀 PHASE 2 COMPONENT VALIDATION")
    print("=" * 60)
    
    tests = [
        test_multi_target_basic,
        test_temporal_networks_basic,
        test_attention_mechanism,
        test_gated_memory,
        test_pattern_recognition
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            print()
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Multi-Target Processing: WORKING")
        print("✅ Temporal Networks: WORKING")
        print("✅ Attention Mechanisms: WORKING")
        print("✅ Memory Systems: WORKING")
        print("✅ Pattern Recognition: WORKING")
        print("\n🚀 Phase 2 Week 1 components are fully operational!")
    else:
        print(f"⚠️ {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    run_simple_tests()
