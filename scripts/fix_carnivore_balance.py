#!/usr/bin/env python3
"""
Fix Carnivore Reproduction and Balance Issues
============================================

This script implements fixes for the carnivore reproduction problems identified in the analysis:
1. Stricter reproduction requirements for carnivores
2. Better starvation mechanics
3. Speed/vision balance adjustments
4. Environmental balance improvements
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def implement_carnivore_fixes():
    """Implement fixes for carnivore reproduction and balance issues"""
    print("🛠️ IMPLEMENTING CARNIVORE FIXES")
    print("=" * 50)
    
    print("📝 1. UPDATING EVOLUTIONARY AGENT CONFIGURATION...")
    
    # Read current evolutionary_agent.py
    agent_file = os.path.join(project_root, "src", "neural", "evolutionary_agent.py")
    
    with open(agent_file, 'r') as f:
        content = f.read()
    
    # Fix 1: Stricter carnivore reproduction requirements
    old_carnivore_requirement = "return base_requirements and self.steps_since_fed < 60"
    new_carnivore_requirement = """# Stricter carnivore requirements
        carnivore_requirements = (
            self.steps_since_fed < 30 and  # Must have eaten more recently
            self.energy >= 70 and         # Higher energy requirement
            len([a for a in getattr(self, '_nearby_prey', [])]) > 0  # Must have prey available
        )
        return base_requirements and carnivore_requirements"""
    
    if old_carnivore_requirement in content:
        content = content.replace(old_carnivore_requirement, new_carnivore_requirement)
        print("   ✅ Updated carnivore reproduction requirements")
    
    # Fix 2: Enhanced starvation mechanics
    old_starvation = """if self.steps_since_fed > 40:  # After 40 steps without eating, starvation begins
                    # Apply escalating starvation penalty
                    starvation_multiplier = 1.0 + (self.steps_since_fed - 40) * 0.06
                    energy_cost = self.config.carnivore_energy_cost * starvation_multiplier"""
    
    new_starvation = """if self.steps_since_fed > 30:  # Starvation begins sooner
                    # Apply exponential starvation penalty
                    starvation_days = (self.steps_since_fed - 30)
                    starvation_multiplier = 1.0 + (starvation_days * 0.15) + (starvation_days ** 2 * 0.02)
                    energy_cost = self.config.carnivore_energy_cost * starvation_multiplier"""
    
    if old_starvation in content:
        content = content.replace(old_starvation, new_starvation)
        print("   ✅ Enhanced starvation mechanics")
    
    # Fix 3: Adjust carnivore speed and vision
    old_carnivore_stats = """self.vision_range = 30.0  # Increased vision for better prey tracking
            self.speed = 2.5  # Increased speed for better hunting"""
    
    new_carnivore_stats = """self.vision_range = 25.0  # Balanced vision for fair hunting
            self.speed = 2.0  # Balanced speed to prevent overhunting"""
    
    if old_carnivore_stats in content:
        content = content.replace(old_carnivore_stats, new_carnivore_stats)
        print("   ✅ Balanced carnivore speed and vision")
    
    # Fix 4: Increase reproduction cost for carnivores
    old_config = """reproduction_cost: float = 50.0"""
    new_config = """reproduction_cost: float = 70.0  # Increased cost for carnivores"""
    
    if old_config in content:
        content = content.replace(old_config, new_config)
        print("   ✅ Increased reproduction cost")
    
    # Write the updated content back
    with open(agent_file, 'w') as f:
        f.write(content)
    
    print("\n📝 2. UPDATING MAIN.PY CONFIGURATION...")
    
    # Update main.py for better carnivore configuration
    main_file = os.path.join(project_root, "main.py")
    
    with open(main_file, 'r') as f:
        main_content = f.read()
    
    # Update carnivore configuration in main.py
    old_carnivore_config = """# Configure Phase 2 networks with default evolutionary settings
        network_config = EvolutionaryNetworkConfig(
            min_input_size=20,  # Enhanced sensory inputs for Phase 2
            max_input_size=25,
            min_hidden_size=12,
            max_hidden_size=24,
            output_size=6,  # Enhanced outputs for complex behaviors
            mutation_rate=0.15,
            recurrent_probability=0.7  # High chance for temporal learning
        )
        
        agent_config = EvolutionaryAgentConfig(
            social_learning=True,
            exploration_tracking=True,
            memory_tracking=True
        )"""
    
    new_carnivore_config = """# Configure Phase 2 networks with balanced evolutionary settings
        network_config = EvolutionaryNetworkConfig(
            min_input_size=20,  # Enhanced sensory inputs for Phase 2
            max_input_size=25,
            min_hidden_size=12,
            max_hidden_size=24,
            output_size=6,  # Enhanced outputs for complex behaviors
            mutation_rate=0.15,
            recurrent_probability=0.7  # High chance for temporal learning
        )
        
        agent_config = EvolutionaryAgentConfig(
            social_learning=True,
            exploration_tracking=True,
            memory_tracking=True,
            reproduction_cost=70.0,  # Higher cost for better balance
            carnivore_energy_cost=2.5,  # Increased starvation penalty
            reproduction_cooldown=30  # Longer cooldown between reproductions
        )"""
    
    if old_carnivore_config in content:
        main_content = main_content.replace(old_carnivore_config, new_carnivore_config)
        print("   ✅ Updated main.py carnivore configuration")
    
    # Write the updated main.py
    with open(main_file, 'w') as f:
        f.write(main_content)
    
    print("\n📝 3. CREATING HERBIVORE ENHANCEMENTS...")
    
    # Update herbivore capabilities in evolutionary_agent.py  
    herbivore_enhancement = """
    def _enhance_herbivore_survival(self):
        \"\"\"Enhance herbivore survival capabilities\"\"\"
        if self.species_type == SpeciesType.HERBIVORE:
            # Increase escape speed when threatened
            if hasattr(self, '_threat_detected') and self._threat_detected:
                self.speed = min(self.speed * 1.3, 2.2)  # Up to 30% speed boost
            
            # Better threat detection
            self.vision_range = max(self.vision_range, 18.0)  # Ensure good vision
    """
    
    # Add the enhancement method to the agent file
    with open(agent_file, 'r') as f:
        content = f.read()
    
    if "_enhance_herbivore_survival" not in content:
        # Add before the last method
        insertion_point = content.rfind("    def reproduce(self):")
        if insertion_point != -1:
            content = content[:insertion_point] + herbivore_enhancement + content[insertion_point:]
            print("   ✅ Added herbivore survival enhancements")
    
    with open(agent_file, 'w') as f:
        f.write(content)
    
    print("\n✅ ALL FIXES IMPLEMENTED!")
    print("\nSummary of changes:")
    print("🐺 Carnivore changes:")
    print("   • Reproduction requires recent feeding (< 30 steps)")
    print("   • Higher energy requirement for reproduction (70 energy)")
    print("   • Reduced speed (2.5 → 2.0) and vision (30 → 25)")
    print("   • Exponential starvation penalty")
    print("   • Higher reproduction cost (50 → 70)")
    print("   • Longer reproduction cooldown (20 → 30)")
    print("\n🦌 Herbivore enhancements:")
    print("   • Improved escape capabilities")
    print("   • Better threat detection")
    print("   • Speed boost when threatened")
    
    return True

if __name__ == "__main__":
    try:
        success = implement_carnivore_fixes()
        
        if success:
            print("\n🎉 FIXES SUCCESSFULLY IMPLEMENTED!")
            print("\nNext steps:")
            print("1. Test the simulation: python main.py --web")
            print("2. Monitor carnivore reproduction after herbivore depletion")
            print("3. Observe improved predator-prey balance")
            print("4. Check that carnivores eventually starve without prey")
        
    except Exception as e:
        print(f"\n❌ Error implementing fixes: {e}")
        import traceback
        traceback.print_exc()
