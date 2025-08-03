#!/usr/bin/env python3
"""
Comprehensive Carnivore Energy Tracking
Track all energy sources for carnivores to find the bug
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType
import time

class EnergyTracker:
    """Track all energy changes for an agent"""
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.energy_log = []
        self.hunt_log = []
        self.food_log = []
        
    def log_energy_change(self, step, old_energy, new_energy, source, details=""):
        """Log an energy change"""
        change = new_energy - old_energy
        entry = {
            'step': step,
            'old_energy': old_energy,
            'new_energy': new_energy,
            'change': change,
            'source': source,
            'details': details
        }
        self.energy_log.append(entry)
        
        if change > 0:
            print(f"   Step {step}: Energy +{change:.1f} from {source} ({details})")
            print(f"      {old_energy:.1f} → {new_energy:.1f}")

class TrackedAgent(NeuralAgent):
    """NeuralAgent with comprehensive energy tracking"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = EnergyTracker(self.id)
        self.step_counter = 0
        
    def successful_hunt(self, energy_gained: int):
        """Override to track hunting"""
        old_energy = self.energy
        super().successful_hunt(energy_gained)
        self.tracker.log_energy_change(
            self.step_counter, old_energy, self.energy, 
            "hunt", f"gained {energy_gained} energy"
        )
        
    def consume_food(self, food_energy: int):
        """Override to track food consumption"""
        old_energy = self.energy
        super().consume_food(food_energy)
        self.tracker.log_energy_change(
            self.step_counter, old_energy, self.energy,
            "food", f"consumed {food_energy} energy"
        )
        
    def update(self):
        """Override to track natural energy loss"""
        old_energy = self.energy
        super().update()
        if self.energy != old_energy:
            self.tracker.log_energy_change(
                self.step_counter, old_energy, self.energy,
                "natural_loss", "metabolic energy cost"
            )
        self.step_counter += 1

def test_comprehensive_energy_tracking():
    """Track all energy sources for carnivores"""
    print("🔍 Comprehensive Carnivore Energy Tracking")
    print("=" * 70)
    
    # Create environment
    env = NeuralEnvironment(width=100, height=100, use_neural_agents=True)
    
    # Replace carnivores with tracked versions
    tracked_carnivores = []
    original_carnivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.CARNIVORE]
    
    for carnivore in original_carnivores:
        # Create tracked version
        tracked = TrackedAgent(
            carnivore.species_type,
            carnivore.position,
            carnivore.id
        )
        tracked.energy = carnivore.energy
        tracked.brain = carnivore.brain
        tracked_carnivores.append(tracked)
        
        # Replace in environment
        env.agents.remove(carnivore)
        env.agents.append(tracked)
    
    # Get counts
    herbivores = [a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE]
    print(f"🦌 Herbivores: {len(herbivores)}")
    print(f"🐺 Tracked Carnivores: {len(tracked_carnivores)}")
    print(f"🌾 Food sources: {len(env.food_sources)}")
    print()
    
    if not tracked_carnivores:
        print("❌ No carnivores to track!")
        return False
        
    # Track specific carnivore
    test_carnivore = tracked_carnivores[0]
    print(f"📊 Tracking Carnivore ID: {test_carnivore.id}")
    print(f"   Initial Energy: {test_carnivore.energy:.1f}")
    print()
    
    # Run simulation and track energy changes
    print("🔄 Running simulation with detailed energy tracking...")
    print("-" * 70)
    
    total_energy_gained = 0
    hunts_observed = 0
    food_consumed = 0
    
    for step in range(50):  # Run longer to see patterns
        step_start_energy = test_carnivore.energy
        herbivore_count_before = len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        
        # Run step
        env.step()
        
        step_end_energy = test_carnivore.energy
        herbivore_count_after = len([a for a in env.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE and a.is_alive])
        
        # Check for energy changes
        energy_change = step_end_energy - step_start_energy
        herbivores_killed = herbivore_count_before - herbivore_count_after
        
        if energy_change > 0:
            total_energy_gained += energy_change
            print(f"Step {step}: Energy gain +{energy_change:.1f} (Total: {step_end_energy:.1f})")
            if herbivores_killed > 0:
                print(f"   🎯 {herbivores_killed} herbivore(s) killed")
                hunts_observed += herbivores_killed
            else:
                print(f"   ❓ Energy gained without visible hunting!")
        
        if not test_carnivore.is_alive:
            print("💀 Carnivore died")
            break
    
    print("\n📈 Energy Analysis:")
    print(f"   Total energy gained: {total_energy_gained:.1f}")
    print(f"   Hunts observed: {hunts_observed}")
    print(f"   Successful hunts recorded: {test_carnivore.lifetime_successful_hunts}")
    print(f"   Food consumed: {test_carnivore.lifetime_food_consumed}")
    
    # Analyze energy log
    print("\n🔬 Detailed Energy Log:")
    for entry in test_carnivore.tracker.energy_log:
        if entry['change'] > 0:
            print(f"   Step {entry['step']}: +{entry['change']:.1f} from {entry['source']} - {entry['details']}")
    
    # Check for anomalies
    energy_from_hunts = sum(entry['change'] for entry in test_carnivore.tracker.energy_log if entry['source'] == 'hunt')
    energy_from_food = sum(entry['change'] for entry in test_carnivore.tracker.energy_log if entry['source'] == 'food')
    energy_from_unknown = total_energy_gained - energy_from_hunts - energy_from_food
    
    print(f"\n🔍 Energy Source Analysis:")
    print(f"   Energy from hunts: {energy_from_hunts:.1f}")
    print(f"   Energy from food: {energy_from_food:.1f}")
    print(f"   Energy from unknown sources: {energy_from_unknown:.1f}")
    
    if energy_from_food > 0:
        print("❌ BUG: Carnivore consumed food!")
        return False
    elif energy_from_unknown > 0:
        print("❌ BUG: Carnivore gained energy from unknown source!")
        return False
    elif energy_from_hunts != total_energy_gained:
        print("❌ BUG: Energy accounting doesn't match!")
        return False
    else:
        print("✅ All energy gains properly accounted for")
        return True

def investigate_hunting_mechanics():
    """Investigate the hunting mechanics in detail"""
    print("\n🎯 Investigating Hunting Mechanics")
    print("=" * 45)
    
    env = NeuralEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Find carnivore and herbivore
    carnivore = None
    herbivore = None
    
    for agent in env.agents:
        if isinstance(agent, NeuralAgent):
            if agent.species_type == SpeciesType.CARNIVORE and not carnivore:
                carnivore = agent
            elif agent.species_type == SpeciesType.HERBIVORE and not herbivore:
                herbivore = agent
    
    if not carnivore or not herbivore:
        print("❌ Need both carnivore and herbivore!")
        return False
        
    # Place them close together
    carnivore.position.x = 25
    carnivore.position.y = 25
    herbivore.position.x = 27
    herbivore.position.y = 26
    
    print(f"🐺 Carnivore: ID {carnivore.id}, Energy {carnivore.energy:.1f}")
    print(f"🦌 Herbivore: ID {herbivore.id}, Energy {herbivore.energy:.1f}")
    print(f"📏 Distance: {carnivore.position.distance_to(herbivore.position):.2f}")
    
    # Test hunting directly
    old_carnivore_energy = carnivore.energy
    old_herbivore_alive = herbivore.is_alive
    
    # Call the feeding handler directly
    env._handle_neural_feeding(carnivore)
    
    energy_gained = carnivore.energy - old_carnivore_energy
    herbivore_killed = old_herbivore_alive and not herbivore.is_alive
    
    print(f"🐺 Carnivore energy after: {carnivore.energy:.1f} (change: {energy_gained:+.1f})")
    print(f"🦌 Herbivore alive: {herbivore.is_alive}")
    
    if energy_gained > 0 and herbivore_killed:
        expected_energy = int(old_carnivore_energy + herbivore.energy * 0.8)
        print(f"✅ Normal hunt: gained {energy_gained:.1f} energy")
        print(f"   Expected energy from prey: {herbivore.energy * 0.8:.1f}")
        return True
    elif energy_gained > 0:
        print(f"❌ BUG: Energy gained without killing prey!")
        return False
    else:
        print("ℹ️ No hunt this attempt (random chance)")
        return True

if __name__ == "__main__":
    try:
        print("🧪 Comprehensive Carnivore Energy Investigation")
        print("=" * 80)
        
        result1 = test_comprehensive_energy_tracking()
        result2 = investigate_hunting_mechanics()
        
        print(f"\n🏁 Final Results:")
        print(f"   Comprehensive Tracking: {'✅ PASS' if result1 else '❌ FAIL'}")
        print(f"   Hunting Mechanics: {'✅ PASS' if result2 else '❌ FAIL'}")
        
        if result1 and result2:
            print("\n🎉 No bugs detected in carnivore energy system!")
        else:
            print("\n⚠️ Energy bugs detected!")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
