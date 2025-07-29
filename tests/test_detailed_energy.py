#!/usr/bin/env python3
"""
Detailed Carnivore Energy Trace
Track every energy change in carnivores step by step
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType

class EnergyTrackingAgent(NeuralAgent):
    """Neural agent with detailed energy tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy_log = []
        self.last_recorded_energy = self.energy
    
    def log_energy_change(self, reason, old_energy, new_energy):
        """Log an energy change with reason"""
        change = new_energy - old_energy
        self.energy_log.append({
            'reason': reason,
            'old_energy': old_energy,
            'new_energy': new_energy,
            'change': change,
            'step': len(self.energy_log)
        })
        print(f"   Energy: {old_energy:.1f} -> {new_energy:.1f} ({change:+.1f}) [{reason}]")
    
    def consume_food(self, food_energy):
        """Track food consumption"""
        old_energy = self.energy
        super().consume_food(food_energy)
        self.log_energy_change("consume_food", old_energy, self.energy)
    
    def successful_hunt(self, energy_gained):
        """Track successful hunts"""
        old_energy = self.energy
        super().successful_hunt(energy_gained)
        self.log_energy_change("successful_hunt", old_energy, self.energy)
    
    def update(self):
        """Track energy decay"""
        old_energy = self.energy
        super().update()
        if self.energy != old_energy:
            self.log_energy_change("energy_decay", old_energy, self.energy)
    
    def check_for_unknown_changes(self):
        """Check if energy changed without logging"""
        if abs(self.energy - self.last_recorded_energy) > 0.001:
            self.log_energy_change("UNKNOWN_CHANGE", self.last_recorded_energy, self.energy)
        self.last_recorded_energy = self.energy

def test_detailed_energy_tracking():
    """Track every energy change in detail"""
    print("🔬 Detailed Carnivore Energy Tracking")
    print("=" * 50)
    
    # Create environment with tracking agents
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Replace carnivores with tracking agents
    tracking_carnivores = []
    for i, agent in enumerate(env.agents):
        if isinstance(agent, NeuralAgent) and agent.species_type == SpeciesType.CARNIVORE:
            # Create tracking version
            tracking_agent = EnergyTrackingAgent(
                agent.species_type, 
                agent.position, 
                agent.id
            )
            tracking_agent.energy = agent.energy
            tracking_agent.age = agent.age
            tracking_agent.last_recorded_energy = agent.energy
            
            # Copy brain if it exists
            if hasattr(agent, 'brain'):
                tracking_agent.brain = agent.brain
            
            # Replace in environment
            env.agents[i] = tracking_agent
            tracking_carnivores.append(tracking_agent)
            
            # Only track first carnivore
            if len(tracking_carnivores) == 1:
                break
    
    if not tracking_carnivores:
        print("❌ No carnivores found!")
        return
    
    test_carnivore = tracking_carnivores[0]
    
    print(f"🎯 Tracking Carnivore ID: {test_carnivore.id}")
    print(f"   Initial Energy: {test_carnivore.energy:.1f}")
    print(f"   Energy Decay: {test_carnivore.energy_decay}")
    print()
    
    # Move carnivore away from all prey
    test_carnivore.position.x = 10
    test_carnivore.position.y = 10
    
    # Move all herbivores away
    for agent in env.agents:
        if hasattr(agent, 'species_type') and agent.species_type == SpeciesType.HERBIVORE:
            agent.position.x = 90
            agent.position.y = 90
    
    print("📍 Carnivore isolated from all prey")
    print()
    
    # Track changes step by step
    for step in range(10):
        print(f"🔄 Step {step + 1}:")
        
        # Check for any unexpected changes before step
        test_carnivore.check_for_unknown_changes()
        
        # Record energy before step
        energy_before = test_carnivore.energy
        
        # Run step
        env.step()
        
        # Check for any unexpected changes after step
        test_carnivore.check_for_unknown_changes()
        
        # Summary
        energy_after = test_carnivore.energy
        total_change = energy_after - energy_before
        print(f"   Total step change: {total_change:+.1f}")
        print()
        
        if not test_carnivore.is_alive:
            print("💀 Carnivore died!")
            break
    
    print("📊 Energy Change Log:")
    for entry in test_carnivore.energy_log:
        print(f"   Step {entry['step']}: {entry['reason']} ({entry['change']:+.1f})")
    
    return True

if __name__ == "__main__":
    try:
        test_detailed_energy_tracking()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
