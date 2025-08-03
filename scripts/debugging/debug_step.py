#!/usr/bin/env python3
"""
Step-by-step debugging of the hunting process
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural.neural_agents import NeuralEnvironment, NeuralAgent
from src.core.ecosystem import SpeciesType
import random

class DebugEnvironment(NeuralEnvironment):
    """Environment with detailed step debugging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_log = []
    
    def debug_step(self):
        """Enhanced step with detailed logging"""
        self.time_step += 1
        new_agents = []
        
        print(f"\n=== STEP {self.time_step} START ===")
        
        # Print initial state
        carnivores = [a for a in self.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.CARNIVORE]
        herbivores = [a for a in self.agents if isinstance(a, NeuralAgent) and a.species_type == SpeciesType.HERBIVORE]
        
        print(f"Initial state:")
        for c in carnivores:
            print(f"  Carnivore {c.id}: Energy {c.energy:.1f}, Hunts {c.lifetime_successful_hunts}")
        
        alive_herbivores = [h for h in herbivores if h.is_alive]
        print(f"  Alive herbivores: {len(alive_herbivores)}")
        for h in alive_herbivores[:3]:  # Show first 3
            print(f"    Herbivore {h.id}: Energy {h.energy:.1f}, Alive {h.is_alive}")
        
        # Update all neural agents
        for i, agent in enumerate(self.agents):
            if not agent.is_alive:
                continue
            
            print(f"\n--- Processing Agent {agent.id} (#{i}) ---")
            
            # Neural decision-making replaces rule-based behavior
            if isinstance(agent, NeuralAgent):
                print(f"Agent {agent.id}: Species {agent.species_type}, Energy {agent.energy:.1f}")
                
                # Reset hunting flag for this step
                agent.was_hunted = False
                print(f"Agent {agent.id}: Reset was_hunted to False")
                
                # Use neural network for decision making
                offspring = agent.neural_move(self)
                if offspring:
                    new_agents.append(offspring)
                    print(f"Agent {agent.id}: Created offspring")
                
                # Handle feeding behavior with neural decisions
                if agent.species_type == SpeciesType.CARNIVORE:
                    print(f"Agent {agent.id}: Handling carnivore feeding...")
                    energy_before = agent.energy
                    hunts_before = agent.lifetime_successful_hunts
                    
                    # Count alive herbivores before
                    herbivores_before = len([h for h in self.agents if isinstance(h, NeuralAgent) and h.species_type == SpeciesType.HERBIVORE and h.is_alive])
                    
                    self._handle_neural_feeding(agent)
                    
                    energy_after = agent.energy
                    hunts_after = agent.lifetime_successful_hunts
                    
                    # Count alive herbivores after
                    herbivores_after = len([h for h in self.agents if isinstance(h, NeuralAgent) and h.species_type == SpeciesType.HERBIVORE and h.is_alive])
                    
                    if energy_after > energy_before:
                        print(f"Agent {agent.id}: ENERGY GAINED! {energy_before:.1f} → {energy_after:.1f}")
                        print(f"Agent {agent.id}: Hunt count: {hunts_before} → {hunts_after}")
                        print(f"Agent {agent.id}: Herbivores alive: {herbivores_before} → {herbivores_after}")
                        
                        if herbivores_after == herbivores_before:
                            print(f"Agent {agent.id}: ❌ BUG: Energy gained but no herbivore died!")
                
                # Update fitness for evolution
                agent.update_fitness(self)
            else:
                # Fallback to original behavior for non-neural agents
                self._handle_traditional_behavior(agent)
            
            # Update agent state
            agent.update()
            
            # Keep agent in bounds
            self.keep_agent_in_bounds(agent)
        
        print(f"\n--- End of agent processing ---")
        
        # Add new offspring
        for new_agent in new_agents:
            self.keep_agent_in_bounds(new_agent)
            self.agents.append(new_agent)
            self.next_agent_id += 1
        
        # Remove dead agents
        dead_agents = [agent for agent in self.agents if not agent.is_alive]
        print(f"Removing {len(dead_agents)} dead agents:")
        for agent in dead_agents:
            print(f"  Removing dead {agent.species_type} {agent.id}")
        
        self.agents = [agent for agent in self.agents if agent.is_alive]
        
        # Trigger evolutionary events during simulation
        self.trigger_evolutionary_events()
        
        # Regenerate food
        for food in self.food_sources:
            if not food.is_available:
                food.current_regen += 1
                if food.current_regen >= food.regeneration_time:
                    food.is_available = True
        
        print(f"=== STEP {self.time_step} END ===")

def debug_single_step():
    """Debug a single step in detail"""
    print("🔍 Debugging Single Step in Detail")
    print("=" * 60)
    
    # Seed for reproducible results
    random.seed(42)
    
    env = DebugEnvironment(width=50, height=50, use_neural_agents=True)
    
    # Find one carnivore and place it near a herbivore
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
        return
    
    # Place them close together
    carnivore.position.x = 25
    carnivore.position.y = 25
    herbivore.position.x = 26
    herbivore.position.y = 25
    
    print(f"🐺 Test Carnivore: {carnivore.id}")
    print(f"🦌 Test Herbivore: {herbivore.id}")
    print(f"📏 Distance: {carnivore.position.distance_to(herbivore.position):.2f}")
    
    # Run one debug step
    env.debug_step()

if __name__ == "__main__":
    try:
        debug_single_step()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
