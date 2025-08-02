"""
Social Learning Framework Tests

Test the multi-channel communication, social influence, and collective intelligence
components of the Phase 2 social learning system.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural.social_learning import (
    SocialLearningFramework, CommunicationChannel, SocialInfluenceModel,
    create_social_learning_framework, simulate_collective_intelligence
)


def test_communication_channel():
    """Test communication channel functionality"""
    print("📡 Testing Communication Channel...")
    
    # Create channel
    channel = CommunicationChannel("test_food", "food", max_range=30.0)
    
    # Test message broadcasting
    food_info = {
        'food_x': 25.0,
        'food_y': 35.0, 
        'food_value': 20.0,
        'urgency': 0.8,
        'confidence': 0.9
    }
    
    message_id = channel.broadcast_message(
        sender_id="agent_1",
        sender_pos=(20.0, 30.0),
        information=food_info,
        timestamp=100
    )
    
    assert message_id is not None
    assert len(channel.active_messages) == 1
    
    # Test message receiving
    received = channel.receive_messages(
        receiver_id="agent_2",
        receiver_pos=(25.0, 35.0),  # Close to sender
        timestamp=101
    )
    
    assert len(received) == 1
    assert received[0]['sender_id'] == "agent_1"
    assert received[0]['channel_type'] == "food"
    assert 'decoded_content' in received[0]
    
    print(f"   ✅ Message broadcast and received successfully")
    print(f"   ✅ Signal strength: {received[0]['signal_strength']:.3f}")
    print(f"   ✅ Information value: {received[0]['information_value']:.3f}")
    
    return True


def test_social_influence_model():
    """Test social influence and reputation system"""
    print("🤝 Testing Social Influence Model...")
    
    # Create model
    model = SocialInfluenceModel("agent_1")
    
    # Test trust updates
    model.update_trust("agent_2", 0.5, 0.8)  # Positive interaction, high accuracy
    model.update_trust("agent_3", -0.3, 0.4)  # Negative interaction, low accuracy
    
    assert model.trust_levels["agent_2"] > 0
    assert model.trust_levels["agent_3"] < 0
    
    # Test influence weight calculation
    influence_2 = model.calculate_influence_weight("agent_2", "food")
    influence_3 = model.calculate_influence_weight("agent_3", "food")
    
    assert influence_2 > influence_3  # More trusted agent has more influence
    
    # Test cooperation decision
    should_cooperate_2 = model.should_cooperate_with("agent_2", 0.3)
    should_cooperate_3 = model.should_cooperate_with("agent_3", 0.3)
    
    print(f"   ✅ Trust levels - Agent 2: {model.trust_levels['agent_2']:.3f}, Agent 3: {model.trust_levels['agent_3']:.3f}")
    print(f"   ✅ Influence weights - Agent 2: {influence_2:.3f}, Agent 3: {influence_3:.3f}")
    print(f"   ✅ Cooperation decisions - Agent 2: {should_cooperate_2}, Agent 3: {should_cooperate_3}")
    
    return True


def test_social_learning_framework():
    """Test complete social learning framework"""
    print("🧠 Testing Social Learning Framework...")
    
    # Create frameworks for multiple agents
    framework_1 = create_social_learning_framework("agent_1", 30.0)
    framework_2 = create_social_learning_framework("agent_2", 30.0)
    framework_3 = create_social_learning_framework("agent_3", 30.0)
    
    frameworks = [framework_1, framework_2, framework_3]
    
    # Agent 1 broadcasts food information
    food_info = {
        'food_x': 40.0,
        'food_y': 50.0,
        'food_value': 25.0,
        'urgency': 0.7,
        'confidence': 0.8
    }
    
    msg_id = framework_1.broadcast_information(
        information_type="food",
        information=food_info,
        agent_pos=(35.0, 45.0),
        timestamp=200
    )
    
    assert msg_id is not None
    
    # Agent 2 receives messages
    received_messages = framework_2.receive_messages(
        agent_pos=(38.0, 47.0),  # Close to agent 1
        timestamp=201,
        other_frameworks=frameworks
    )
    
    assert len(received_messages) > 0
    
    # Process social learning
    current_knowledge = {'known_food_sources': []}
    updated_knowledge = framework_2.process_social_learning(
        received_messages, current_knowledge
    )
    
    assert 'social_learning_updates' in updated_knowledge
    assert len(updated_knowledge['known_food_sources']) > 0
    
    print(f"   ✅ Message broadcasting working")
    print(f"   ✅ Messages received: {len(received_messages)}")
    print(f"   ✅ Social learning processed: {len(updated_knowledge['social_learning_updates'])} updates")
    print(f"   ✅ Food sources learned: {len(updated_knowledge['known_food_sources'])}")
    
    return True


def test_multi_channel_communication():
    """Test multiple communication channels"""
    print("📻 Testing Multi-Channel Communication...")
    
    framework = create_social_learning_framework("test_agent", 30.0)
    
    # Test different message types
    messages = [
        ("food", {'food_x': 30, 'food_y': 40, 'food_value': 15}),
        ("danger", {'threat_x': 60, 'threat_y': 70, 'danger_level': 8}),
        ("exploration", {'area_x': 50, 'area_y': 50, 'novelty_score': 0.9}),
        ("social", {'cooperation_request': 0.8, 'trust_level': 0.7})
    ]
    
    message_ids = []
    for msg_type, info in messages:
        msg_id = framework.broadcast_information(
            information_type=msg_type,
            information=info,
            agent_pos=(45.0, 45.0),
            timestamp=300
        )
        message_ids.append(msg_id)
    
    # Verify all messages were broadcast
    assert all(msg_id is not None for msg_id in message_ids)
    
    # Check channel statistics
    stats = framework.get_framework_stats()
    assert 'communication_channels' in stats
    assert len(stats['communication_channels']) == 4
    
    print(f"   ✅ All 4 channel types working")
    print(f"   ✅ Messages broadcast: {len([mid for mid in message_ids if mid])}")
    print(f"   ✅ Channel statistics: {list(stats['communication_channels'].keys())}")
    
    return True


def test_collective_intelligence():
    """Test collective intelligence emergence"""
    print("🧬 Testing Collective Intelligence...")
    
    # Create multiple agents
    frameworks = []
    for i in range(5):
        fw = create_social_learning_framework(f"agent_{i}", 25.0)
        frameworks.append(fw)
    
    # Each agent shares different information
    for i, framework in enumerate(frameworks):
        # Share food information
        food_info = {
            'food_x': 20 + i * 15,
            'food_y': 30 + i * 10,
            'food_value': 10 + i * 5,
            'confidence': 0.7 + i * 0.05
        }
        
        framework.broadcast_information(
            "food", food_info, (20 + i * 10, 30 + i * 10), 400 + i
        )
        
        # Add some shared knowledge
        framework.shared_knowledge['known_food_sources'] = [
            {'x': food_info['food_x'], 'y': food_info['food_y'], 'value': food_info['food_value']}
        ]
        framework.shared_knowledge['explored_areas'] = {f"area_{i}": {'resource_density': 0.5 + i * 0.1}}
    
    # Simulate collective intelligence
    collective_metrics = simulate_collective_intelligence(frameworks, 405)
    
    assert 'total_food_knowledge' in collective_metrics
    assert 'communication_efficiency' in collective_metrics
    assert 'social_cohesion' in collective_metrics
    
    print(f"   ✅ Food knowledge aggregated: {collective_metrics['total_food_knowledge']} sources")
    print(f"   ✅ Area coverage: {collective_metrics['explored_area_coverage']} areas")
    print(f"   ✅ Communication efficiency: {collective_metrics['communication_efficiency']:.3f}")
    print(f"   ✅ Social cohesion: {collective_metrics['social_cohesion']:.3f}")
    
    return True


def test_trust_and_reputation():
    """Test trust and reputation system"""
    print("⭐ Testing Trust and Reputation System...")
    
    framework_1 = create_social_learning_framework("agent_1", 30.0)
    framework_2 = create_social_learning_framework("agent_2", 30.0)
    
    # Simulate interactions with different outcomes
    interactions = [
        {'other_agent_id': 'agent_2', 'outcome': 0.8, 'information_accuracy': 0.9},  # Good
        {'other_agent_id': 'agent_2', 'outcome': 0.6, 'information_accuracy': 0.8},  # Good
        {'other_agent_id': 'agent_2', 'outcome': -0.2, 'information_accuracy': 0.3}, # Bad
        {'other_agent_id': 'agent_2', 'outcome': 0.7, 'information_accuracy': 0.85}, # Good
    ]
    
    # Update relationships
    framework_1.update_social_relationships(interactions)
    
    # Check trust level
    trust_level = framework_1.social_model.trust_levels['agent_2']
    social_stats = framework_1.social_model.get_social_stats()
    
    assert trust_level > 0  # Should be positive overall
    assert social_stats['total_interactions'] == 4
    
    # Test cooperation decision
    should_cooperate = framework_1.social_model.should_cooperate_with('agent_2', 0.3)
    
    print(f"   ✅ Trust level after interactions: {trust_level:.3f}")
    print(f"   ✅ Total interactions: {social_stats['total_interactions']}")
    print(f"   ✅ Success rate: {social_stats['success_rate']:.3f}")
    print(f"   ✅ Cooperation decision: {should_cooperate}")
    
    return True


def run_social_learning_tests():
    """Run all social learning tests"""
    print("=" * 70)
    print("🤝 SOCIAL LEARNING FRAMEWORK TESTS")
    print("=" * 70)
    
    tests = [
        test_communication_channel,
        test_social_influence_model,
        test_social_learning_framework,
        test_multi_channel_communication,
        test_collective_intelligence,
        test_trust_and_reputation
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
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 70)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL SOCIAL LEARNING TESTS PASSED!")
        print("✅ Multi-Channel Communication: WORKING")
        print("✅ Social Influence & Trust: WORKING")
        print("✅ Reputation System: WORKING")
        print("✅ Collective Intelligence: WORKING") 
        print("✅ Information Sharing: WORKING")
        print("✅ Social Coordination: WORKING")
        print("\n🚀 Social Learning Framework is fully operational!")
    else:
        print(f"⚠️ {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    run_social_learning_tests()
