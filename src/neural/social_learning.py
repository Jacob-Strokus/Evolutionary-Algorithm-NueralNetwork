"""
Social Learning Framework for Phase 2 Evolutionary Neural Networks

This module implements advanced communication and cooperation mechanisms enabling:
- Multi-channel communication protocols
- Information value assessment  
- Social hierarchy emergence
- Collective intelligence patterns

Key Features:
- Dynamic communication networks between agents
- Information sharing with value assessment
- Social influence modeling and reputation systems
- Collective decision making and knowledge transfer
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import math
from collections import defaultdict, deque
import json

class CommunicationChannel:
    """
    Individual communication channel for specific types of information
    """
    
    def __init__(self, channel_id: str, channel_type: str, max_range: float = 30.0):
        """
        Initialize communication channel
        
        Args:
            channel_id: Unique identifier for this channel
            channel_type: Type of information (food, danger, exploration, social)
            max_range: Maximum communication range
        """
        self.channel_id = channel_id
        self.channel_type = channel_type
        self.max_range = max_range
        
        # Message buffer and history
        self.active_messages = []
        self.message_history = deque(maxlen=100)
        
        # Channel statistics
        self.total_messages = 0
        self.successful_transmissions = 0
        self.information_value_sum = 0.0
        
        # Encoding/decoding parameters
        self.encoding_weights = np.random.normal(0, 0.1, (8, 16))  # 8 input -> 16 encoded
        self.decoding_weights = np.random.normal(0, 0.1, (16, 8))  # 16 encoded -> 8 output
        
    def encode_message(self, raw_information: Dict) -> np.ndarray:
        """
        Encode raw information into neural message format
        
        Args:
            raw_information: Dictionary with information to encode
            
        Returns:
            Encoded message vector
        """
        # Convert information to feature vector
        feature_vector = self._extract_features(raw_information)
        
        # Neural encoding
        encoded = np.tanh(np.dot(feature_vector, self.encoding_weights))
        
        return encoded
    
    def decode_message(self, encoded_message: np.ndarray) -> Dict:
        """
        Decode neural message back to information
        
        Args:
            encoded_message: Encoded message vector
            
        Returns:
            Decoded information dictionary
        """
        # Neural decoding
        decoded_features = np.tanh(np.dot(encoded_message, self.decoding_weights))
        
        # Convert back to information structure
        information = self._reconstruct_information(decoded_features)
        
        return information
    
    def broadcast_message(self, sender_id: str, sender_pos: Tuple[float, float], 
                         information: Dict, timestamp: int) -> str:
        """
        Broadcast a message on this channel
        
        Args:
            sender_id: ID of sending agent
            sender_pos: Position of sender
            information: Information to broadcast
            timestamp: Current simulation timestamp
            
        Returns:
            Message ID
        """
        # Encode the message
        encoded_msg = self.encode_message(information)
        
        # Calculate information value
        info_value = self._assess_information_value(information)
        
        # Create message object
        message = {
            'id': f"{self.channel_id}_{timestamp}_{len(self.active_messages)}",
            'sender_id': sender_id,
            'sender_pos': sender_pos,
            'timestamp': timestamp,
            'channel_type': self.channel_type,
            'encoded_content': encoded_msg,
            'raw_content': information,
            'information_value': info_value,
            'received_by': set(),
            'transmission_range': self.max_range
        }
        
        # Add to active messages
        self.active_messages.append(message)
        self.message_history.append(message.copy())
        
        # Update statistics
        self.total_messages += 1
        self.information_value_sum += info_value
        
        return message['id']
    
    def receive_messages(self, receiver_id: str, receiver_pos: Tuple[float, float], 
                        timestamp: int) -> List[Dict]:
        """
        Receive messages within range for a specific agent
        
        Args:
            receiver_id: ID of receiving agent
            receiver_pos: Position of receiver
            timestamp: Current simulation timestamp
            
        Returns:
            List of received messages
        """
        received_messages = []
        
        for message in self.active_messages:
            # Check if already received
            if receiver_id in message['received_by']:
                continue
            
            # Check range
            distance = self._calculate_distance(receiver_pos, message['sender_pos'])
            if distance > message['transmission_range']:
                continue
            
            # Check temporal validity (messages decay over time)
            age = timestamp - message['timestamp']
            if age > 20:  # Messages expire after 20 time steps
                continue
            
            # Signal strength based on distance
            signal_strength = max(0.1, 1.0 - (distance / message['transmission_range']))
            
            # Add noise based on signal strength
            noisy_content = self._add_transmission_noise(
                message['encoded_content'], signal_strength
            )
            
            # Create received message
            received_msg = {
                'message_id': message['id'],
                'sender_id': message['sender_id'],
                'channel_type': message['channel_type'],
                'encoded_content': noisy_content,
                'signal_strength': signal_strength,
                'age': age,
                'information_value': message['information_value'] * signal_strength
            }
            
            # Decode content
            try:
                received_msg['decoded_content'] = self.decode_message(noisy_content)
            except:
                received_msg['decoded_content'] = {}
            
            received_messages.append(received_msg)
            
            # Mark as received
            message['received_by'].add(receiver_id)
            self.successful_transmissions += 1
        
        return received_messages
    
    def _extract_features(self, information: Dict) -> np.ndarray:
        """Extract neural features from information dictionary"""
        features = np.zeros(8)
        
        if self.channel_type == 'food':
            features[0] = information.get('food_x', 0) / 100.0  # Normalized position
            features[1] = information.get('food_y', 0) / 100.0
            features[2] = information.get('food_value', 0) / 50.0  # Normalized value
            features[3] = information.get('urgency', 0.5)  # Urgency level
            features[4] = information.get('confidence', 0.5)  # Sender confidence
            
        elif self.channel_type == 'danger':
            features[0] = information.get('threat_x', 0) / 100.0
            features[1] = information.get('threat_y', 0) / 100.0
            features[2] = information.get('danger_level', 0) / 10.0
            features[3] = information.get('urgency', 1.0)  # Danger is urgent
            features[4] = information.get('confidence', 0.5)
            features[5] = information.get('escape_direction_x', 0)
            features[6] = information.get('escape_direction_y', 0)
            
        elif self.channel_type == 'exploration':
            features[0] = information.get('area_x', 0) / 100.0
            features[1] = information.get('area_y', 0) / 100.0
            features[2] = information.get('resource_density', 0)
            features[3] = information.get('novelty_score', 0)
            features[4] = information.get('exploration_value', 0)
            
        elif self.channel_type == 'social':
            features[0] = information.get('cooperation_request', 0)
            features[1] = information.get('social_rank', 0.5)
            features[2] = information.get('trust_level', 0.5)
            features[3] = information.get('group_coordination', 0)
            features[4] = information.get('leadership_signal', 0)
        
        return features
    
    def _reconstruct_information(self, features: np.ndarray) -> Dict:
        """Reconstruct information dictionary from neural features"""
        information = {}
        
        if self.channel_type == 'food':
            information = {
                'food_x': features[0] * 100.0,
                'food_y': features[1] * 100.0,
                'food_value': features[2] * 50.0,
                'urgency': features[3],
                'confidence': features[4]
            }
            
        elif self.channel_type == 'danger':
            information = {
                'threat_x': features[0] * 100.0,
                'threat_y': features[1] * 100.0,
                'danger_level': features[2] * 10.0,
                'urgency': features[3],
                'confidence': features[4],
                'escape_direction_x': features[5],
                'escape_direction_y': features[6]
            }
            
        elif self.channel_type == 'exploration':
            information = {
                'area_x': features[0] * 100.0,
                'area_y': features[1] * 100.0,
                'resource_density': features[2],
                'novelty_score': features[3],
                'exploration_value': features[4]
            }
            
        elif self.channel_type == 'social':
            information = {
                'cooperation_request': features[0],
                'social_rank': features[1],
                'trust_level': features[2],
                'group_coordination': features[3],
                'leadership_signal': features[4]
            }
        
        return information
    
    def _assess_information_value(self, information: Dict) -> float:
        """Assess the value of information being transmitted"""
        base_value = 0.5
        
        if self.channel_type == 'food':
            # Higher value for larger food sources and urgent information
            food_value = information.get('food_value', 10) / 50.0
            urgency = information.get('urgency', 0.5)
            confidence = information.get('confidence', 0.5)
            base_value = (food_value + urgency + confidence) / 3.0
            
        elif self.channel_type == 'danger':
            # Higher value for more dangerous threats
            danger_level = information.get('danger_level', 5) / 10.0
            urgency = information.get('urgency', 1.0)
            confidence = information.get('confidence', 0.5)
            base_value = (danger_level + urgency + confidence) / 3.0
            
        elif self.channel_type == 'exploration':
            # Higher value for novel, resource-rich areas
            novelty = information.get('novelty_score', 0.5)
            resources = information.get('resource_density', 0.5)
            exploration_value = information.get('exploration_value', 0.5)
            base_value = (novelty + resources + exploration_value) / 3.0
            
        elif self.channel_type == 'social':
            # Higher value for cooperation and coordination
            cooperation = information.get('cooperation_request', 0)
            coordination = information.get('group_coordination', 0)
            leadership = information.get('leadership_signal', 0)
            base_value = (cooperation + coordination + leadership) / 3.0
        
        return np.clip(base_value, 0.0, 1.0)
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _add_transmission_noise(self, encoded_message: np.ndarray, signal_strength: float) -> np.ndarray:
        """Add noise to message based on signal strength"""
        noise_level = 0.1 * (1.0 - signal_strength)
        noise = np.random.normal(0, noise_level, encoded_message.shape)
        return encoded_message + noise
    
    def cleanup_old_messages(self, current_timestamp: int):
        """Remove expired messages"""
        self.active_messages = [
            msg for msg in self.active_messages 
            if current_timestamp - msg['timestamp'] <= 20
        ]
    
    def get_channel_stats(self) -> Dict:
        """Get channel performance statistics"""
        transmission_rate = (self.successful_transmissions / max(1, self.total_messages))
        avg_info_value = (self.information_value_sum / max(1, self.total_messages))
        
        return {
            'channel_id': self.channel_id,
            'channel_type': self.channel_type,
            'total_messages': self.total_messages,
            'successful_transmissions': self.successful_transmissions,
            'transmission_rate': transmission_rate,
            'avg_information_value': avg_info_value,
            'active_messages': len(self.active_messages)
        }


class SocialInfluenceModel:
    """
    Models social influence and reputation systems between agents
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize social influence model
        
        Args:
            agent_id: ID of the agent this model belongs to
        """
        self.agent_id = agent_id
        
        # Social relationships
        self.trust_levels = defaultdict(float)  # Trust in other agents
        self.influence_weights = defaultdict(float)  # How much others influence this agent
        self.reputation_scores = defaultdict(float)  # Reputation of other agents
        
        # Social learning parameters
        self.learning_rate = 0.1
        self.trust_decay = 0.95
        self.influence_threshold = 0.3
        
        # Social memory
        self.interaction_history = deque(maxlen=200)
        self.successful_collaborations = defaultdict(int)
        self.failed_collaborations = defaultdict(int)
        
        # Social roles and hierarchy
        self.social_rank = 0.5  # 0.0 = follower, 1.0 = leader
        self.leadership_experience = 0.0
        self.cooperation_tendency = 0.5
        
    def update_trust(self, other_agent_id: str, interaction_outcome: float, 
                    information_accuracy: float = None):
        """
        Update trust level with another agent based on interaction outcome
        
        Args:
            other_agent_id: ID of the other agent
            interaction_outcome: Outcome of interaction (-1.0 to 1.0)
            information_accuracy: Accuracy of information provided (0.0 to 1.0)
        """
        current_trust = self.trust_levels[other_agent_id]
        
        # Base trust update from interaction outcome
        trust_change = self.learning_rate * interaction_outcome
        
        # Additional trust change from information accuracy
        if information_accuracy is not None:
            accuracy_bonus = self.learning_rate * (information_accuracy - 0.5) * 2.0
            trust_change += accuracy_bonus
        
        # Update trust with bounds
        new_trust = np.clip(current_trust + trust_change, -1.0, 1.0)
        self.trust_levels[other_agent_id] = new_trust
        
        # Update interaction history
        self.interaction_history.append({
            'other_agent': other_agent_id,
            'outcome': interaction_outcome,
            'accuracy': information_accuracy,
            'trust_before': current_trust,
            'trust_after': new_trust,
            'timestamp': len(self.interaction_history)
        })
        
        # Update collaboration counters
        if interaction_outcome > 0:
            self.successful_collaborations[other_agent_id] += 1
        else:
            self.failed_collaborations[other_agent_id] += 1
    
    def calculate_influence_weight(self, other_agent_id: str, message_type: str) -> float:
        """
        Calculate how much this agent should be influenced by another agent's message
        
        Args:
            other_agent_id: ID of the influencing agent
            message_type: Type of message being received
            
        Returns:
            Influence weight (0.0 to 1.0)
        """
        base_trust = self.trust_levels[other_agent_id]
        reputation = self.reputation_scores[other_agent_id]
        
        # Base influence from trust and reputation
        base_influence = (base_trust + 1.0) / 2.0  # Convert -1,1 to 0,1
        reputation_influence = (reputation + 1.0) / 2.0
        
        # Message type modifiers
        type_modifiers = {
            'danger': 1.2,  # Danger messages are more influential
            'food': 1.0,
            'exploration': 0.8,
            'social': 0.9
        }
        
        type_modifier = type_modifiers.get(message_type, 1.0)
        
        # Social hierarchy influence
        other_rank = self.get_perceived_social_rank(other_agent_id)
        hierarchy_influence = other_rank * 0.3  # Higher rank = more influence
        
        # Calculate final influence weight
        influence_weight = (
            base_influence * 0.4 +
            reputation_influence * 0.3 +
            hierarchy_influence * 0.3
        ) * type_modifier
        
        return np.clip(influence_weight, 0.0, 1.0)
    
    def update_reputation(self, other_agent_id: str, observed_behavior: Dict):
        """
        Update reputation of another agent based on observed behavior
        
        Args:
            other_agent_id: ID of the agent being observed
            observed_behavior: Dictionary with behavior observations
        """
        current_reputation = self.reputation_scores[other_agent_id]
        
        # Extract behavior signals
        cooperation_level = observed_behavior.get('cooperation', 0.5)
        information_sharing = observed_behavior.get('information_sharing', 0.5)
        group_benefit = observed_behavior.get('group_benefit', 0.5)
        leadership_quality = observed_behavior.get('leadership', 0.5)
        
        # Calculate reputation change
        reputation_factors = [cooperation_level, information_sharing, group_benefit, leadership_quality]
        behavior_score = np.mean(reputation_factors)
        
        reputation_change = self.learning_rate * (behavior_score - 0.5) * 2.0
        
        # Update reputation
        new_reputation = np.clip(current_reputation + reputation_change, -1.0, 1.0)
        self.reputation_scores[other_agent_id] = new_reputation
    
    def get_perceived_social_rank(self, other_agent_id: str) -> float:
        """
        Get perceived social rank of another agent
        
        Args:
            other_agent_id: ID of the other agent
            
        Returns:
            Perceived social rank (0.0 to 1.0)
        """
        # Base rank from reputation and trust
        reputation = self.reputation_scores[other_agent_id]
        trust = self.trust_levels[other_agent_id]
        
        # Success rate in collaborations
        successes = self.successful_collaborations[other_agent_id]
        failures = self.failed_collaborations[other_agent_id]
        total_interactions = successes + failures
        
        if total_interactions > 0:
            success_rate = successes / total_interactions
        else:
            success_rate = 0.5
        
        # Calculate perceived rank
        perceived_rank = (
            (reputation + 1.0) / 2.0 * 0.4 +
            (trust + 1.0) / 2.0 * 0.3 +
            success_rate * 0.3
        )
        
        return np.clip(perceived_rank, 0.0, 1.0)
    
    def should_cooperate_with(self, other_agent_id: str, cooperation_cost: float) -> bool:
        """
        Decide whether to cooperate with another agent
        
        Args:
            other_agent_id: ID of potential cooperation partner
            cooperation_cost: Cost of cooperation (0.0 to 1.0)
            
        Returns:
            Whether to cooperate
        """
        trust = self.trust_levels[other_agent_id]
        reputation = self.reputation_scores[other_agent_id]
        
        # Calculate cooperation benefit expectation
        expected_benefit = (
            (trust + 1.0) / 2.0 * 0.6 +
            (reputation + 1.0) / 2.0 * 0.4
        )
        
        # Factor in agent's cooperation tendency
        cooperation_probability = (
            expected_benefit * 0.7 +
            self.cooperation_tendency * 0.3
        )
        
        # Consider cost
        net_benefit = cooperation_probability - cooperation_cost
        
        return net_benefit > 0.2  # Cooperation threshold
    
    def decay_social_memory(self):
        """Apply decay to trust and reputation over time"""
        for agent_id in list(self.trust_levels.keys()):
            self.trust_levels[agent_id] *= self.trust_decay
            
        for agent_id in list(self.reputation_scores.keys()):
            self.reputation_scores[agent_id] *= self.trust_decay
    
    def get_social_stats(self) -> Dict:
        """Get social influence statistics"""
        total_trust = sum(self.trust_levels.values())
        num_relationships = len(self.trust_levels)
        avg_trust = total_trust / max(1, num_relationships)
        
        total_reputation = sum(self.reputation_scores.values())
        avg_reputation = total_reputation / max(1, len(self.reputation_scores))
        
        total_interactions = sum(self.successful_collaborations.values()) + sum(self.failed_collaborations.values())
        success_rate = sum(self.successful_collaborations.values()) / max(1, total_interactions)
        
        return {
            'agent_id': self.agent_id,
            'social_rank': self.social_rank,
            'cooperation_tendency': self.cooperation_tendency,
            'num_relationships': num_relationships,
            'avg_trust': avg_trust,
            'avg_reputation': avg_reputation,
            'total_interactions': total_interactions,
            'success_rate': success_rate,
            'leadership_experience': self.leadership_experience
        }


class SocialLearningFramework:
    """
    Main social learning framework combining communication and influence systems
    """
    
    def __init__(self, agent_id: str, communication_range: float = 30.0):
        """
        Initialize social learning framework
        
        Args:
            agent_id: ID of the agent
            communication_range: Default communication range
        """
        self.agent_id = agent_id
        self.communication_range = communication_range
        
        # Communication channels
        self.channels = {
            'food': CommunicationChannel(f"{agent_id}_food", 'food', communication_range),
            'danger': CommunicationChannel(f"{agent_id}_danger", 'danger', communication_range * 1.5),  # Danger travels further
            'exploration': CommunicationChannel(f"{agent_id}_exploration", 'exploration', communication_range),
            'social': CommunicationChannel(f"{agent_id}_social", 'social', communication_range * 0.8)  # Social is more local
        }
        
        # Social influence model
        self.social_model = SocialInfluenceModel(agent_id)
        
        # Collective intelligence
        self.shared_knowledge = {}
        self.collective_decisions = []
        
        # Learning and adaptation
        self.message_effectiveness = defaultdict(list)
        self.communication_preferences = defaultdict(float)
        
    def broadcast_information(self, information_type: str, information: Dict, 
                            agent_pos: Tuple[float, float], timestamp: int) -> str:
        """
        Broadcast information on appropriate channel
        
        Args:
            information_type: Type of information (food, danger, exploration, social)
            information: Information content
            agent_pos: Current agent position
            timestamp: Current simulation timestamp
            
        Returns:
            Message ID if successful, None if failed
        """
        if information_type not in self.channels:
            return None
        
        channel = self.channels[information_type]
        message_id = channel.broadcast_message(self.agent_id, agent_pos, information, timestamp)
        
        # Track communication preferences
        self.communication_preferences[information_type] += 0.1
        
        return message_id
    
    def receive_messages(self, agent_pos: Tuple[float, float], timestamp: int, 
                        other_frameworks: List['SocialLearningFramework']) -> List[Dict]:
        """
        Receive messages from all channels and other agents
        
        Args:
            agent_pos: Current agent position
            timestamp: Current simulation timestamp
            other_frameworks: List of other agents' social learning frameworks
            
        Returns:
            List of received messages with influence weights
        """
        all_received_messages = []
        
        # Receive from all channels of all agents
        for other_framework in other_frameworks:
            if other_framework.agent_id == self.agent_id:
                continue
                
            for channel_type, channel in other_framework.channels.items():
                messages = channel.receive_messages(self.agent_id, agent_pos, timestamp)
                
                for message in messages:
                    # Calculate influence weight
                    influence_weight = self.social_model.calculate_influence_weight(
                        message['sender_id'], channel_type
                    )
                    
                    message['influence_weight'] = influence_weight
                    message['should_act_on'] = influence_weight > self.social_model.influence_threshold
                    
                    all_received_messages.append(message)
        
        return all_received_messages
    
    def process_social_learning(self, received_messages: List[Dict], 
                               current_knowledge: Dict) -> Dict:
        """
        Process received messages for social learning
        
        Args:
            received_messages: Messages received from other agents
            current_knowledge: Agent's current knowledge state
            
        Returns:
            Updated knowledge incorporating social learning
        """
        updated_knowledge = current_knowledge.copy()
        learning_updates = []
        
        for message in received_messages:
            if not message['should_act_on']:
                continue
            
            influence_weight = message['influence_weight']
            decoded_content = message['decoded_content']
            
            # Process different types of social learning
            if message['channel_type'] == 'food':
                self._process_food_learning(decoded_content, influence_weight, updated_knowledge)
                
            elif message['channel_type'] == 'danger':
                self._process_danger_learning(decoded_content, influence_weight, updated_knowledge)
                
            elif message['channel_type'] == 'exploration':
                self._process_exploration_learning(decoded_content, influence_weight, updated_knowledge)
                
            elif message['channel_type'] == 'social':
                self._process_social_coordination(decoded_content, influence_weight, updated_knowledge)
            
            # Track learning
            learning_updates.append({
                'source': message['sender_id'],
                'type': message['channel_type'],
                'influence': influence_weight,
                'content': decoded_content
            })
        
        # Update shared knowledge
        updated_knowledge['social_learning_updates'] = learning_updates
        updated_knowledge['collective_knowledge'] = self.shared_knowledge
        
        return updated_knowledge
    
    def _process_food_learning(self, food_info: Dict, influence_weight: float, knowledge: Dict):
        """Process food-related social learning"""
        if 'known_food_sources' not in knowledge:
            knowledge['known_food_sources'] = []
        
        # Add or update food source information
        food_location = (food_info.get('food_x', 0), food_info.get('food_y', 0))
        food_value = food_info.get('food_value', 0) * influence_weight
        
        # Update or add food source
        updated = False
        for i, food_source in enumerate(knowledge['known_food_sources']):
            if abs(food_source['x'] - food_location[0]) < 5 and abs(food_source['y'] - food_location[1]) < 5:
                # Update existing source
                knowledge['known_food_sources'][i]['value'] = max(
                    food_source['value'], food_value
                )
                knowledge['known_food_sources'][i]['confidence'] += influence_weight * 0.1
                updated = True
                break
        
        if not updated:
            knowledge['known_food_sources'].append({
                'x': food_location[0],
                'y': food_location[1],
                'value': food_value,
                'confidence': influence_weight
            })
    
    def _process_danger_learning(self, danger_info: Dict, influence_weight: float, knowledge: Dict):
        """Process danger-related social learning"""
        if 'known_threats' not in knowledge:
            knowledge['known_threats'] = []
        
        threat_location = (danger_info.get('threat_x', 0), danger_info.get('threat_y', 0))
        danger_level = danger_info.get('danger_level', 0) * influence_weight
        
        # Add threat information
        knowledge['known_threats'].append({
            'x': threat_location[0],
            'y': threat_location[1],
            'danger_level': danger_level,
            'escape_x': danger_info.get('escape_direction_x', 0),
            'escape_y': danger_info.get('escape_direction_y', 0),
            'urgency': danger_info.get('urgency', 1.0) * influence_weight
        })
        
        # Keep only recent threats
        if len(knowledge['known_threats']) > 10:
            knowledge['known_threats'] = knowledge['known_threats'][-10:]
    
    def _process_exploration_learning(self, exploration_info: Dict, influence_weight: float, knowledge: Dict):
        """Process exploration-related social learning"""
        if 'explored_areas' not in knowledge:
            knowledge['explored_areas'] = {}
        
        area_key = f"{int(exploration_info.get('area_x', 0)/10)}_{int(exploration_info.get('area_y', 0)/10)}"
        
        if area_key not in knowledge['explored_areas']:
            knowledge['explored_areas'][area_key] = {
                'resource_density': 0,
                'novelty_score': 0,
                'exploration_value': 0,
                'confidence': 0
            }
        
        # Update area information
        area = knowledge['explored_areas'][area_key]
        area['resource_density'] = max(area['resource_density'], 
                                     exploration_info.get('resource_density', 0) * influence_weight)
        area['novelty_score'] = max(area['novelty_score'],
                                  exploration_info.get('novelty_score', 0) * influence_weight)
        area['exploration_value'] += exploration_info.get('exploration_value', 0) * influence_weight * 0.1
        area['confidence'] = min(1.0, area['confidence'] + influence_weight * 0.1)
    
    def _process_social_coordination(self, social_info: Dict, influence_weight: float, knowledge: Dict):
        """Process social coordination messages"""
        if 'social_coordination' not in knowledge:
            knowledge['social_coordination'] = {
                'cooperation_requests': [],
                'group_decisions': [],
                'leadership_signals': []
            }
        
        # Process cooperation requests
        if social_info.get('cooperation_request', 0) > 0.5:
            knowledge['social_coordination']['cooperation_requests'].append({
                'request_strength': social_info['cooperation_request'] * influence_weight,
                'trust_level': social_info.get('trust_level', 0.5),
                'influence_weight': influence_weight
            })
        
        # Process leadership signals
        if social_info.get('leadership_signal', 0) > 0.5:
            knowledge['social_coordination']['leadership_signals'].append({
                'leadership_strength': social_info['leadership_signal'] * influence_weight,
                'social_rank': social_info.get('social_rank', 0.5),
                'influence_weight': influence_weight
            })
    
    def update_social_relationships(self, interaction_results: List[Dict]):
        """
        Update social relationships based on interaction results
        
        Args:
            interaction_results: List of interaction outcome dictionaries
        """
        for result in interaction_results:
            other_agent_id = result['other_agent_id']
            outcome = result['outcome']  # -1.0 to 1.0
            information_accuracy = result.get('information_accuracy')
            
            # Update trust
            self.social_model.update_trust(other_agent_id, outcome, information_accuracy)
            
            # Update reputation if behavior observed
            if 'observed_behavior' in result:
                self.social_model.update_reputation(other_agent_id, result['observed_behavior'])
    
    def cleanup_old_data(self, current_timestamp: int):
        """Clean up old messages and data"""
        for channel in self.channels.values():
            channel.cleanup_old_messages(current_timestamp)
        
        # Apply social memory decay
        self.social_model.decay_social_memory()
    
    def get_framework_stats(self) -> Dict:
        """Get comprehensive social learning statistics"""
        channel_stats = {}
        for channel_type, channel in self.channels.items():
            channel_stats[channel_type] = channel.get_channel_stats()
        
        social_stats = self.social_model.get_social_stats()
        
        return {
            'agent_id': self.agent_id,
            'communication_channels': channel_stats,
            'social_influence': social_stats,
            'communication_preferences': dict(self.communication_preferences),
            'shared_knowledge_size': len(self.shared_knowledge),
            'collective_decisions': len(self.collective_decisions)
        }


# Utility functions for integration
def create_social_learning_framework(agent_id: str, communication_range: float = 30.0) -> SocialLearningFramework:
    """Factory function to create social learning framework"""
    return SocialLearningFramework(agent_id, communication_range)


def simulate_collective_intelligence(frameworks: List[SocialLearningFramework], 
                                  timestamp: int) -> Dict:
    """
    Simulate collective intelligence emergence across multiple agents
    
    Args:
        frameworks: List of social learning frameworks
        timestamp: Current simulation timestamp
        
    Returns:
        Collective intelligence metrics
    """
    # Gather collective knowledge
    all_food_sources = []
    all_threats = []
    all_explored_areas = {}
    
    for framework in frameworks:
        knowledge = framework.shared_knowledge
        
        # Aggregate food sources
        if 'known_food_sources' in knowledge:
            all_food_sources.extend(knowledge['known_food_sources'])
        
        # Aggregate threats
        if 'known_threats' in knowledge:
            all_threats.extend(knowledge['known_threats'])
        
        # Aggregate exploration data
        if 'explored_areas' in knowledge:
            for area, data in knowledge['explored_areas'].items():
                if area not in all_explored_areas:
                    all_explored_areas[area] = data.copy()
                else:
                    # Combine exploration data
                    all_explored_areas[area]['resource_density'] = max(
                        all_explored_areas[area]['resource_density'],
                        data['resource_density']
                    )
    
    # Calculate collective intelligence metrics
    collective_metrics = {
        'timestamp': timestamp,
        'total_food_knowledge': len(all_food_sources),
        'total_threat_awareness': len(all_threats),
        'explored_area_coverage': len(all_explored_areas),
        'knowledge_redundancy': _calculate_knowledge_redundancy(frameworks),
        'communication_efficiency': _calculate_communication_efficiency(frameworks),
        'social_cohesion': _calculate_social_cohesion(frameworks)
    }
    
    return collective_metrics


def _calculate_knowledge_redundancy(frameworks: List[SocialLearningFramework]) -> float:
    """Calculate knowledge redundancy across agents"""
    if len(frameworks) < 2:
        return 0.0
    
    # Compare knowledge overlap
    overlaps = []
    for i, fw1 in enumerate(frameworks):
        for j, fw2 in enumerate(frameworks[i+1:], i+1):
            overlap = _calculate_knowledge_overlap(fw1.shared_knowledge, fw2.shared_knowledge)
            overlaps.append(overlap)
    
    return np.mean(overlaps) if overlaps else 0.0


def _calculate_knowledge_overlap(knowledge1: Dict, knowledge2: Dict) -> float:
    """Calculate overlap between two knowledge bases"""
    # Simple implementation - can be expanded
    common_keys = set(knowledge1.keys()) & set(knowledge2.keys())
    total_keys = set(knowledge1.keys()) | set(knowledge2.keys())
    
    if not total_keys:
        return 0.0
    
    return len(common_keys) / len(total_keys)


def _calculate_communication_efficiency(frameworks: List[SocialLearningFramework]) -> float:
    """Calculate overall communication efficiency"""
    if not frameworks:
        return 0.0
    
    total_efficiency = 0.0
    for framework in frameworks:
        channel_stats = framework.get_framework_stats()['communication_channels']
        
        for channel_type, stats in channel_stats.items():
            total_efficiency += stats['transmission_rate']
    
    return total_efficiency / (len(frameworks) * 4)  # 4 channels per agent


def _calculate_social_cohesion(frameworks: List[SocialLearningFramework]) -> float:
    """Calculate social cohesion of the group"""
    if len(frameworks) < 2:
        return 0.0
    
    trust_levels = []
    for framework in frameworks:
        social_stats = framework.get_framework_stats()['social_influence']
        trust_levels.append(social_stats['avg_trust'])
    
    return np.mean(trust_levels)
