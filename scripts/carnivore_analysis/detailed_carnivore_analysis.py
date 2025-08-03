#!/usr/bin/env python3
"""
Detailed carnivore behavior analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import Phase2NeuralEnvironment
from src.core.ecosystem import SpeciesType

def detailed_carnivore_analysis():
    """Analyze carnivore behavior in detail"""
    print("🔬 Detailed Carnivore Starvation Analysis")
    print("=" * 50)
    
    env = Phase2NeuralEnvironment(100, 100)
    
    # Remove all herbivores to force starvation scenario
    env.agents = [a for a in env.agents if a.species_type == SpeciesType.CARNIVORE]
    initial_count = len(env.agents)
    
    print(f"🏁 Starting with {initial_count} carnivores (herbivores removed)")
    
    # Track specific carnivore over time
    if env.agents:
        tracked_carnivore = env.agents[0]
        print(f"📊 Tracking Carnivore ID: {tracked_carnivore.agent_id}")
        print(f"   Initial Energy: {tracked_carnivore.energy}")
        print(f"   Max Energy: {tracked_carnivore.max_energy}")
        print(f"   Energy Decay Config: {tracked_carnivore.config.age_energy_cost}")
        print(f"   Carnivore Energy Cost: {tracked_carnivore.config.carnivore_energy_cost}")
        print(f"   Steps Since Fed: {tracked_carnivore.steps_since_fed}")
    
    print(f"\n📈 Step-by-Step Analysis:")
    
    for step in range(300):
        prev_count = len(env.agents)
        env.step()
        new_count = len(env.agents)
        
        if step % 25 == 0 or step < 10:
            avg_energy = sum(a.energy for a in env.agents) / len(env.agents) if env.agents else 0
            avg_steps_since_fed = sum(getattr(a, 'steps_since_fed', 0) for a in env.agents) / len(env.agents) if env.agents else 0
            alive_count = len([a for a in env.agents if a.is_alive])
            
            print(f"   Step {step:3d}: {alive_count:2d} alive, {new_count:2d} total, "
                  f"avg energy: {avg_energy:5.1f}, avg starved: {avg_steps_since_fed:5.1f}")
            
            # Show details of tracked carnivore
            if env.agents and tracked_carnivore in env.agents:
                tc = tracked_carnivore
                can_repro = tc.can_reproduce()
                print(f"            Tracked: Energy={tc.energy:.1f}, StarvedSteps={tc.steps_since_fed}, "
                      f"CanReproduce={can_repro}, Age={tc.age}")
        
        # Show population changes
        if new_count != prev_count:
            change = new_count - prev_count
            print(f"   Step {step:3d}: Population change: {change:+d} ({'births' if change > 0 else 'deaths'})")
    
    final_count = len([a for a in env.agents if a.is_alive])
    print(f"\n📊 Final Results:")
    print(f"   Started: {initial_count} carnivores")
    print(f"   Final: {final_count} carnivores")
    print(f"   Change: {final_count - initial_count:+d} ({((final_count - initial_count) / initial_count * 100):+.1f}%)")

if __name__ == "__main__":
    detailed_carnivore_analysis()
