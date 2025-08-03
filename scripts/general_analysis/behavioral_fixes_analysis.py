"""
Ecosystem Behavioral Fixes
Addresses carnivore extinction and herbivore boundary issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import math
from src.core.ecosystem import Position

def analyze_ecosystem_issues():
    """Analyze the identified ecosystem behavioral issues"""
    
    print("🔍 ECOSYSTEM BEHAVIORAL ANALYSIS")
    print("=" * 60)
    
    issues = {
        "carnivore_extinction": {
            "description": "Carnivores die out quickly",
            "causes": [
                "Herbivores run to edges, making hunting difficult",
                "Limited prey availability due to boundary clustering", 
                "Energy costs of hunting may be too high",
                "Carnivore reproduction requirements too strict"
            ],
            "severity": "HIGH"
        },
        "herbivore_boundary_seeking": {
            "description": "Herbivores cluster at boundaries and starve",
            "causes": [
                "Boundary avoidance signals may be inverted", 
                "Food distribution concentrated in center",
                "Neural network interpreting boundary distance incorrectly",
                "Edge-seeking behavior being rewarded in fitness"
            ],
            "severity": "HIGH"
        },
        "population_explosion": {
            "description": "Herbivore population grows rapidly then crashes",
            "causes": [
                "No predation pressure after carnivore extinction",
                "Abundant food sources with fast regeneration",
                "Low energy cost for reproduction",
                "Poor resource competition mechanics"
            ],
            "severity": "MEDIUM"
        }
    }
    
    print("🚨 IDENTIFIED ISSUES:")
    for issue_name, issue_data in issues.items():
        print(f"\n📌 {issue_name.replace('_', ' ').title()}")
        print(f"   Severity: {issue_data['severity']}")
        print(f"   Description: {issue_data['description']}")
        print("   Potential Causes:")
        for cause in issue_data['causes']:
            print(f"     • {cause}")
    
    return issues

def create_behavioral_fixes():
    """Create fixes for the behavioral issues"""
    
    print("\n🔧 PROPOSED FIXES")
    print("=" * 40)
    
    fixes = {
        "boundary_behavior": {
            "description": "Fix boundary avoidance behavior",
            "changes": [
                "Invert boundary distance signals (0=edge, 1=center)",
                "Add strong fitness penalty for boundary proximity", 
                "Improve neural network boundary input interpretation",
                "Add center-seeking reward bias"
            ]
        },
        "carnivore_survival": {
            "description": "Improve carnivore viability",
            "changes": [
                "Reduce carnivore energy decay rate",
                "Increase hunt success probability",
                "Lower carnivore reproduction threshold",
                "Add group hunting mechanics"
            ]
        },
        "food_distribution": {
            "description": "Better food placement strategy", 
            "changes": [
                "Distribute food away from boundaries",
                "Create food clusters to encourage foraging",
                "Reduce food regeneration rate slightly",
                "Add food scarcity pressure"
            ]
        },
        "fitness_rebalancing": {
            "description": "Rebalance fitness incentives",
            "changes": [
                "Strong penalty for boundary proximity",
                "Reward center area occupation", 
                "Increase carnivore hunt rewards",
                "Balance reproduction costs"
            ]
        }
    }
    
    for fix_name, fix_data in fixes.items():
        print(f"\n🛠️  {fix_name.replace('_', ' ').title()}")
        print(f"   Description: {fix_data['description']}")
        print("   Proposed Changes:")
        for change in fix_data['changes']:
            print(f"     ✅ {change}")
    
    return fixes

def create_enhanced_ecosystem_fixes():
    """Generate code for the enhanced ecosystem fixes"""
    
    enhanced_ecosystem_code = '''
"""
Enhanced Ecosystem with Behavioral Fixes
Addresses carnivore extinction and herbivore boundary issues
"""

class EnhancedBehavioralFixes:
    """Behavioral fixes for ecosystem issues"""
    
    @staticmethod
    def enhanced_boundary_fitness_penalty(agent, environment):
        """Apply strong penalty for boundary proximity"""
        # Calculate distance to nearest boundary
        x_dist_to_edge = min(agent.position.x, environment.width - agent.position.x)
        y_dist_to_edge = min(agent.position.y, environment.height - agent.position.y)
        min_boundary_distance = min(x_dist_to_edge, y_dist_to_edge)
        
        # Normalize to 0-1 (0 = at edge, 1 = in center)
        boundary_ratio = min_boundary_distance / min(environment.width, environment.height) * 2
        
        # Strong penalty for being near boundaries
        if boundary_ratio < 0.1:  # Very close to edge
            return -50.0  # Severe penalty
        elif boundary_ratio < 0.2:  # Close to edge  
            return -20.0  # Moderate penalty
        elif boundary_ratio > 0.5:  # In center area
            return +5.0   # Center bonus
        else:
            return 0.0    # Neutral
    
    @staticmethod
    def enhanced_carnivore_survival_boost(carnivore):
        """Boost carnivore survival chances"""
        survival_boosts = {
            'energy_decay_reduction': 0.7,    # 30% less energy loss
            'hunt_success_bonus': 0.3,        # 30% better hunt chance
            'reproduction_threshold': 120,     # Lower reproduction requirement
            'vision_range_bonus': 1.2         # 20% better vision
        }
        return survival_boosts
    
    @staticmethod
    def enhanced_food_distribution(environment):
        """Create better food distribution avoiding boundaries"""
        # Clear existing food
        environment.food_sources = []
        
        # Create food clusters away from boundaries
        num_clusters = 3
        cluster_size = 4
        
        for cluster in range(num_clusters):
            # Place cluster center away from boundaries  
            center_x = random.uniform(environment.width * 0.3, environment.width * 0.7)
            center_y = random.uniform(environment.height * 0.3, environment.height * 0.7)
            
            # Create cluster of food around center
            for food_item in range(cluster_size):
                offset_x = random.uniform(-15, 15)
                offset_y = random.uniform(-15, 15)
                
                food_x = max(20, min(environment.width - 20, center_x + offset_x))
                food_y = max(20, min(environment.height - 20, center_y + offset_y))
                
                from src.core.ecosystem import Food, Position
                food = Food(Position(food_x, food_y))
                food.regeneration_time = 150  # Slower regeneration
                environment.food_sources.append(food)
    
    @staticmethod
    def enhanced_neural_boundary_inputs(agent, environment):
        """Enhanced boundary input calculation for neural networks"""
        # Distance to edges (inverted: 0=edge, 1=center)
        x_center_ratio = 1.0 - (2 * abs(agent.position.x - environment.width/2) / environment.width)
        y_center_ratio = 1.0 - (2 * abs(agent.position.y - environment.height/2) / environment.height)
        
        # Boundary proximity warning (higher = closer to boundary)
        x_boundary_warning = 1.0 if min(agent.position.x, environment.width - agent.position.x) < 20 else 0.0
        y_boundary_warning = 1.0 if min(agent.position.y, environment.height - agent.position.y) < 20 else 0.0
        
        return {
            'x_center_ratio': max(0.0, min(1.0, x_center_ratio)),
            'y_center_ratio': max(0.0, min(1.0, y_center_ratio)),
            'x_boundary_warning': x_boundary_warning,
            'y_boundary_warning': y_boundary_warning
        }
'''
    
    return enhanced_ecosystem_code

if __name__ == "__main__":
    issues = analyze_ecosystem_issues()
    fixes = create_behavioral_fixes()
    enhanced_code = create_enhanced_ecosystem_fixes()
    
    print("\n📝 IMPLEMENTATION PLAN:")
    print("-" * 30)
    print("1. Update neural network boundary inputs (invert signals)")
    print("2. Add strong boundary fitness penalties")  
    print("3. Enhance carnivore survival parameters")
    print("4. Improve food distribution strategy")
    print("5. Test with enhanced evolution demo")
    
    print("\n🎯 EXPECTED IMPROVEMENTS:")
    print("-" * 30)
    print("✅ Herbivores avoid boundaries and forage in center")
    print("✅ Carnivores survive longer with better hunting")
    print("✅ Balanced population dynamics")
    print("✅ Stable ecosystem without crashes")
    
    print(f"\n💾 Enhanced code ready for implementation")
    print("Next: Apply fixes to neural network and ecosystem files")
