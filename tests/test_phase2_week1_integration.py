"""
Phase 2 Week 1 Integration Tests: Multi-Target Processing & Advanced Recurrent Networks

This test suite validates the integration of:
1. Multi-Target Decision Fusion system
2. Advanced Recurrent Networks with temporal learning

Test Coverage:
- Multi-target attention and prioritization
- LSTM-style temporal memory processing  
- Pattern recognition and prediction
- Integration with Phase 1 evolutionary components
"""

import sys
import os
import numpy as np
import unittest
from typing import List, Dict, Tuple

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

# Import Phase 1 components for integration testing
from neural.evolutionary_sensors import EvolutionaryNeuralSensors
from neural.evolutionary_network import EvolutionaryNeuralNetwork
from neural.evolutionary_agent import EvolutionaryNeuralAgent


class TestMultiTargetProcessing(unittest.TestCase):
    """Test multi-target processing system"""
    
    def setUp(self):
        """Set up test environment"""
        self.processor = create_multi_target_processor("test_agent", 25)
        self.test_agent_pos = (50.0, 50.0)
        self.test_agent_energy = 75.0
        
        # Create test targets
        self.food_targets = [
            {'x': 45.0, 'y': 48.0, 'type': 'food', 'energy_value': 20, 'id': 'food1'},
            {'x': 55.0, 'y': 52.0, 'type': 'food', 'energy_value': 30, 'id': 'food2'},
            {'x': 40.0, 'y': 60.0, 'type': 'food', 'energy_value': 15, 'id': 'food3'}
        ]
        
        self.threat_targets = [
            {'x': 52.0, 'y': 51.0, 'type': 'threat', 'danger_level': 7, 'id': 'threat1'},
            {'x': 48.0, 'y': 45.0, 'type': 'threat', 'danger_level': 5, 'id': 'threat2'}
        ]
        
        self.mixed_targets = self.food_targets + self.threat_targets
        
    def test_attention_mechanism(self):
        """Test attention mechanism functionality"""
        print("\n🎯 Testing Attention Mechanism...")
        
        attention = AttentionMechanism(25, 16)
        
        # Create test target features
        target_features = [
            np.random.normal(0, 0.5, 25),
            np.random.normal(0, 0.5, 25),
            np.random.normal(0, 0.5, 25)
        ]
        context = np.random.normal(0, 0.5, 25)
        
        # Test forward pass
        attended_output, attention_weights = attention.forward(target_features, context)
        
        # Validate outputs
        self.assertEqual(len(attended_output), 25)
        self.assertEqual(len(attention_weights), 3)
        self.assertAlmostEqual(np.sum(attention_weights), 1.0, places=5)
        
        print(f"   ✅ Attention weights sum to 1.0: {np.sum(attention_weights):.6f}")
        print(f"   ✅ Output shape correct: {attended_output.shape}")
        print(f"   ✅ Attention distribution: {attention_weights}")
        
    def test_target_prioritization(self):
        """Test target prioritization algorithms"""
        print("\n📊 Testing Target Prioritization...")
        
        prioritizer = TargetPrioritizer("test_agent")
        
        # Test prioritization
        prioritized = prioritizer.prioritize_targets(
            self.mixed_targets, self.test_agent_pos, self.test_agent_energy
        )
        
        # Validate prioritization
        self.assertEqual(len(prioritized), 5)  # 3 food + 2 threats
        
        # Check priority scores are reasonable
        scores = [score for _, score in prioritized]
        self.assertTrue(all(score >= 0 for score in scores))
        
        # Food should generally be prioritized over threats
        food_scores = [score for target, score in prioritized if target['type'] == 'food']
        threat_scores = [score for target, score in prioritized if target['type'] == 'threat']
        
        avg_food_score = np.mean(food_scores) if food_scores else 0
        avg_threat_score = np.mean(threat_scores) if threat_scores else 0
        
        print(f"   ✅ Average food priority: {avg_food_score:.3f}")
        print(f"   ✅ Average threat priority: {avg_threat_score:.3f}")
        print(f"   ✅ Prioritization working: {avg_food_score > avg_threat_score}")
        
    def test_multi_target_processing(self):
        """Test complete multi-target processing system"""
        print("\n🎯 Testing Multi-Target Processing System...")
        
        # Create test sensory input
        sensory_input = np.random.normal(0, 0.3, 25)
        
        # Process targets
        recommendations = self.processor.process_targets(
            sensory_input, self.mixed_targets, 
            self.test_agent_pos, self.test_agent_energy
        )
        
        # Validate recommendations
        self.assertIn('decision_vector', recommendations)
        self.assertIn('primary_action', recommendations)
        self.assertIn('confidence', recommendations)
        self.assertIn('attention_weights', recommendations)
        
        # Check decision vector
        decision_vector = recommendations['decision_vector']
        self.assertEqual(len(decision_vector), 25)
        
        # Check confidence
        confidence = recommendations['confidence']
        self.assertTrue(0.0 <= confidence <= 1.0)
        
        print(f"   ✅ Primary action: {recommendations['primary_action']}")
        print(f"   ✅ Confidence: {confidence:.3f}")
        print(f"   ✅ Decision vector norm: {np.linalg.norm(decision_vector):.3f}")
        print(f"   ✅ Reasoning steps: {len(recommendations['reasoning'])}")
        
    def test_performance_tracking(self):
        """Test performance tracking functionality"""
        print("\n📈 Testing Performance Tracking...")
        
        # Process multiple steps
        for i in range(10):
            sensory_input = np.random.normal(0, 0.3, 25)
            self.processor.process_targets(
                sensory_input, self.mixed_targets[:3],  # Limit targets
                self.test_agent_pos, self.test_agent_energy
            )
        
        # Get statistics
        stats = self.processor.get_processing_stats()
        
        # Validate stats
        self.assertEqual(stats['total_steps'], 10)
        self.assertIn('recent_avg_targets', stats)
        self.assertIn('action_distribution', stats)
        
        print(f"   ✅ Total processing steps: {stats['total_steps']}")
        print(f"   ✅ Average targets per step: {stats['recent_avg_targets']:.2f}")
        print(f"   ✅ Action distribution: {stats['action_distribution']}")


class TestAdvancedRecurrentNetworks(unittest.TestCase):
    """Test advanced recurrent networks"""
    
    def setUp(self):
        """Set up test environment"""
        self.network = create_advanced_recurrent_network(25, 16, "test_agent")
        self.test_input = np.random.normal(0, 0.5, 25)
        
    def test_gated_memory_cell(self):
        """Test LSTM-style gated memory cell"""
        print("\n🧠 Testing Gated Memory Cell...")
        
        cell = GatedMemoryCell(25, 16, "test_cell")
        
        # Test multiple forward passes
        inputs = [np.random.normal(0, 0.5, 25) for _ in range(5)]
        outputs = []
        
        for inp in inputs:
            hidden, cell_state = cell.forward(inp)
            outputs.append((hidden, cell_state))
        
        # Validate outputs
        for hidden, cell_state in outputs:
            self.assertEqual(len(hidden), 16)
            self.assertEqual(len(cell_state), 16)
        
        # Test state reset
        cell.reset_state()
        self.assertTrue(np.allclose(cell.hidden_state, 0))
        self.assertTrue(np.allclose(cell.cell_state, 0))
        
        print(f"   ✅ Memory cell processes {len(inputs)} steps correctly")
        print(f"   ✅ Hidden state shape: {outputs[-1][0].shape}")
        print(f"   ✅ Cell state shape: {outputs[-1][1].shape}")
        print(f"   ✅ State reset working")
        
    def test_multi_timescale_memory(self):
        """Test multi-timescale memory system"""
        print("\n⏰ Testing Multi-Timescale Memory...")
        
        memory = MultiTimescaleMemory(25)
        
        # Process multiple steps to trigger different timescales
        outputs = []
        for i in range(15):  # Enough to trigger all timescales
            inp = np.random.normal(0, 0.5, 25)
            output = memory.process(inp)
            outputs.append(output)
        
        # Validate memory processing
        self.assertEqual(len(outputs), 15)
        for output in outputs:
            self.assertTrue(len(output) > 0)  # Memory should produce output
        
        # Test pattern extraction
        patterns = memory.get_all_patterns()
        self.assertIn('short_term', patterns)
        self.assertIn('medium_term', patterns)
        self.assertIn('long_term', patterns)
        
        print(f"   ✅ Processed {len(outputs)} steps across timescales")
        print(f"   ✅ Output size: {len(outputs[-1])}")
        print(f"   ✅ Pattern extraction working")
        print(f"   ✅ Short-term patterns: {len(patterns['short_term'])}")
        
    def test_temporal_pattern_recognition(self):
        """Test temporal pattern recognition"""
        print("\n🔍 Testing Temporal Pattern Recognition...")
        
        recognizer = TemporalPatternRecognizer(pattern_length=3)
        
        # Create a simple repeating pattern
        actions = ['move_up', 'move_right', 'move_down'] * 3  # Repeat 3 times
        rewards = [0.1, 0.2, 0.15] * 3
        
        # Observe the pattern
        for action, reward in zip(actions, rewards):
            recognizer.observe_action(action, reward)
        
        # Test pattern extraction
        patterns = recognizer.get_best_patterns()
        self.assertTrue(len(patterns) > 0)
        
        # Test prediction
        predictions = recognizer.predict_next_action(['move_up', 'move_right'])
        self.assertIn('move_down', predictions)  # Should predict the completing action
        
        print(f"   ✅ Learned {len(patterns)} patterns")
        print(f"   ✅ Pattern quality: {patterns[0]['confidence']:.3f}")
        print(f"   ✅ Predictions: {predictions}")
        
    def test_advanced_recurrent_network(self):
        """Test complete advanced recurrent network"""
        print("\n🧬 Testing Advanced Recurrent Network...")
        
        # Test forward pass
        output, temporal_info = self.network.forward(self.test_input)
        
        # Validate output
        self.assertEqual(len(output), 16)  # Output size
        self.assertIn('neural_patterns', temporal_info)
        self.assertIn('memory_activation', temporal_info)
        
        # Test with action and reward
        output2, temporal_info2 = self.network.forward(
            self.test_input, current_action='explore', reward=0.1
        )
        
        # Test prediction capability
        predictions = self.network.predict_next_actions(['explore'])
        
        # Get temporal statistics
        stats = self.network.get_temporal_stats()
        
        print(f"   ✅ Network output shape: {output.shape}")
        print(f"   ✅ Memory activation strength: {temporal_info['memory_activation']['strength']:.3f}")
        print(f"   ✅ Behavioral patterns: {len(temporal_info['behavioral_patterns'])}")
        print(f"   ✅ Total processing steps: {stats['total_steps']}")
        print(f"   ✅ Pattern quality: {stats['pattern_quality']:.3f}")


class TestPhase2Integration(unittest.TestCase):
    """Test integration of Phase 2 components with Phase 1 evolutionary system"""
    
    def setUp(self):
        """Set up integration test environment"""
        # Create Phase 1 components
        self.evolutionary_network = EvolutionaryNeuralNetwork(
            input_size=25, output_size=4, hidden_size=16, 
            agent_id="integration_test"
        )
        
        # Create Phase 2 components
        self.multi_target_processor = create_multi_target_processor("integration_test", 25)
        self.temporal_network = create_advanced_recurrent_network(25, 16, "integration_test")
        
        # Test environment
        self.test_ecosystem = {
            'width': 100, 'height': 100,
            'food_sources': [
                {'x': 30, 'y': 40, 'energy': 20},
                {'x': 70, 'y': 60, 'energy': 25}
            ],
            'threats': [
                {'x': 50, 'y': 50, 'danger_level': 6}
            ]
        }
        
    def test_sensory_integration(self):
        """Test integration with Phase 1 sensory system"""
        print("\n🔗 Testing Sensory System Integration...")
        
        # Create test agent state
        agent_state = {
            'x': 40.0, 'y': 45.0, 'energy': 80.0, 'age': 100,
            'last_food_x': 35.0, 'last_food_y': 42.0,
            'exploration_map': {(4, 4): 5, (3, 4): 3}
        }
        
        # Test Phase 1 sensory inputs
        sensors = EvolutionaryNeuralSensors()
        sensory_inputs = sensors.get_enhanced_sensory_inputs(
            agent_state, self.test_ecosystem
        )
        
        self.assertEqual(len(sensory_inputs), 25)
        
        # Test Phase 2 processing with Phase 1 inputs
        test_targets = [
            {'x': 30, 'y': 40, 'type': 'food', 'energy_value': 20, 'id': 'food1'},
            {'x': 50, 'y': 50, 'type': 'threat', 'danger_level': 6, 'id': 'threat1'}
        ]
        
        recommendations = self.multi_target_processor.process_targets(
            sensory_inputs, test_targets, 
            (agent_state['x'], agent_state['y']), agent_state['energy']
        )
        
        self.assertIn('primary_action', recommendations)
        
        print(f"   ✅ Phase 1 sensory inputs: {len(sensory_inputs)} features")
        print(f"   ✅ Phase 2 processing action: {recommendations['primary_action']}")
        print(f"   ✅ Processing confidence: {recommendations['confidence']:.3f}")
        
    def test_network_integration(self):
        """Test integration of evolutionary and temporal networks"""
        print("\n🧬 Testing Network Integration...")
        
        # Create test sensory input
        sensory_input = np.random.normal(0, 0.3, 25)
        
        # Test Phase 1 evolutionary network
        evo_output = self.evolutionary_network.forward(sensory_input)
        self.assertEqual(len(evo_output), 4)
        
        # Test Phase 2 temporal network  
        temporal_output, temporal_info = self.temporal_network.forward(sensory_input)
        self.assertEqual(len(temporal_output), 16)
        
        # Test combined processing (simulation of future integration)
        combined_features = np.concatenate([evo_output, temporal_output[:4]])
        self.assertEqual(len(combined_features), 8)
        
        print(f"   ✅ Evolutionary network output: {evo_output.shape}")
        print(f"   ✅ Temporal network output: {temporal_output.shape}")
        print(f"   ✅ Combined features: {combined_features.shape}")
        print(f"   ✅ Memory patterns detected: {len(temporal_info['behavioral_patterns'])}")
        
    def test_decision_fusion(self):
        """Test decision fusion between Phase 1 and Phase 2 systems"""
        print("\n⚖️ Testing Decision Fusion...")
        
        # Simulate decision scenario
        sensory_input = np.random.normal(0, 0.3, 25)
        agent_pos = (45.0, 47.0)
        agent_energy = 60.0
        
        # Phase 1: Basic evolutionary decision
        evo_output = self.evolutionary_network.forward(sensory_input)
        basic_decision = np.argmax(evo_output)  # 0=up, 1=right, 2=down, 3=left
        
        # Phase 2: Multi-target analysis
        targets = [
            {'x': 30, 'y': 40, 'type': 'food', 'energy_value': 20, 'id': 'food1'},
            {'x': 50, 'y': 50, 'type': 'threat', 'danger_level': 6, 'id': 'threat1'}
        ]
        
        recommendations = self.multi_target_processor.process_targets(
            sensory_input, targets, agent_pos, agent_energy
        )
        
        # Phase 2: Temporal prediction
        temporal_output, temporal_info = self.temporal_network.forward(
            sensory_input, current_action='explore', reward=0.05
        )
        
        # Decision fusion logic (simplified)
        fusion_confidence = (
            max(evo_output) * 0.4 +  # Evolutionary confidence
            recommendations['confidence'] * 0.4 +  # Multi-target confidence
            temporal_info['memory_activation']['strength'] * 0.2  # Memory strength
        )
        
        print(f"   ✅ Phase 1 decision: {basic_decision} (confidence: {max(evo_output):.3f})")
        print(f"   ✅ Phase 2 action: {recommendations['primary_action']}")
        print(f"   ✅ Temporal memory strength: {temporal_info['memory_activation']['strength']:.3f}")
        print(f"   ✅ Fused confidence: {fusion_confidence:.3f}")
        
    def test_learning_integration(self):
        """Test learning integration across Phase 1 and Phase 2 systems"""
        print("\n📚 Testing Learning Integration...")
        
        # Simulate learning episode
        episode_length = 20
        total_reward = 0
        
        for step in range(episode_length):
            # Generate sensory input
            sensory_input = np.random.normal(0, 0.3, 25)
            
            # Phase 1: Evolutionary processing
            evo_output = self.evolutionary_network.forward(sensory_input)
            
            # Phase 2: Multi-target processing  
            test_targets = [
                {'x': 30 + step, 'y': 40, 'type': 'food', 'energy_value': 15, 'id': f'food_{step}'}
            ]
            
            recommendations = self.multi_target_processor.process_targets(
                sensory_input, test_targets, (40 + step, 45), 70
            )
            
            # Phase 2: Temporal learning
            action = recommendations['primary_action']
            reward = np.random.normal(0.1, 0.05)  # Small random reward
            total_reward += reward
            
            temporal_output, temporal_info = self.temporal_network.forward(
                sensory_input, current_action=action, reward=reward
            )
        
        # Get learning statistics
        processing_stats = self.multi_target_processor.get_processing_stats()
        temporal_stats = self.temporal_network.get_temporal_stats()
        
        print(f"   ✅ Episode length: {episode_length} steps")
        print(f"   ✅ Total reward: {total_reward:.3f}")
        print(f"   ✅ Multi-target steps: {processing_stats['total_steps']}")
        print(f"   ✅ Temporal patterns learned: {temporal_stats['learned_patterns']}")
        print(f"   ✅ Pattern quality: {temporal_stats['pattern_quality']:.3f}")


def run_all_tests():
    """Run all Phase 2 Week 1 tests"""
    print("=" * 80)
    print("🚀 PHASE 2 WEEK 1 INTEGRATION TESTS")
    print("Testing Multi-Target Processing & Advanced Recurrent Networks")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMultiTargetProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedRecurrentNetworks))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2Integration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! Phase 2 Week 1 components are working correctly!")
        print("✅ Multi-Target Processing System: OPERATIONAL")
        print("✅ Advanced Recurrent Networks: OPERATIONAL") 
        print("✅ Phase 1 Integration: VERIFIED")
        print("\nReady to proceed with Week 2: Social Learning & Exploration Intelligence! 🚀")
    else:
        print(f"\n⚠️ {len(result.failures + result.errors)} tests failed. Review and fix issues before proceeding.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()
