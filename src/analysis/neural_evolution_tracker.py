"""
Neural Network Evolution Tracker
Advanced metrics and visualizations for tracking neural network learning progress
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # WSL-friendly backend
import numpy as np
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from src.neural.neural_agents import NeuralAgent, NeuralEnvironment
from src.core.ecosystem import SpeciesType
import time

@dataclass
class GenerationMetrics:
    """Metrics for a single generation"""
    generation: int
    step_range: tuple  # (start_step, end_step)
    
    # Population stats
    herbivore_count: int
    carnivore_count: int
    
    # Fitness metrics
    avg_herbivore_fitness: float
    avg_carnivore_fitness: float
    max_herbivore_fitness: float
    max_carnivore_fitness: float
    min_herbivore_fitness: float
    min_carnivore_fitness: float
    
    # Behavioral metrics
    avg_decisions_per_agent: float
    total_reproductions: int
    total_hunts: int
    total_food_consumed: int
    
    # Neural network complexity
    avg_weight_magnitude: float
    weight_diversity: float  # Standard deviation of weights
    
    # Survival metrics
    avg_lifespan: float
    extinction_events: int

class NeuralEvolutionTracker:
    """Track and visualize neural network learning over generations"""
    
    def __init__(self):
        self.generation_metrics: List[GenerationMetrics] = []
        self.current_generation = 0
        self.generation_start_step = 0
        
        # Real-time tracking
        self.fitness_history = {"herbivore": [], "carnivore": []}
        self.behavior_history = []
        self.weight_evolution = []
        
        # Performance baselines
        self.initial_baseline = None
        self.improvement_threshold = 1.1  # 10% improvement
    
    def start_generation(self, step: int):
        """Mark the start of a new generation"""
        self.current_generation += 1
        self.generation_start_step = step
        print(f"\n🧬 Generation {self.current_generation} Starting at Step {step}")
    
    def record_step_metrics(self, env: NeuralEnvironment):
        """Record metrics for current simulation step"""
        neural_agents = [agent for agent in env.agents if isinstance(agent, NeuralAgent)]
        
        if not neural_agents:
            return
        
        # Fitness tracking
        herbivores = [a for a in neural_agents if a.species_type == SpeciesType.HERBIVORE]
        carnivores = [a for a in neural_agents if a.species_type == SpeciesType.CARNIVORE]
        
        if herbivores:
            herb_fitness = [a.brain.fitness_score for a in herbivores]
            self.fitness_history["herbivore"].append({
                'step': env.time_step,
                'avg': np.mean(herb_fitness),
                'max': np.max(herb_fitness),
                'count': len(herbivores)
            })
        
        if carnivores:
            carn_fitness = [a.brain.fitness_score for a in carnivores]
            self.fitness_history["carnivore"].append({
                'step': env.time_step,
                'avg': np.mean(carn_fitness),
                'max': np.max(carn_fitness),
                'count': len(carnivores)
            })
    
    def end_generation(self, env: NeuralEnvironment, end_step: int):
        """Complete generation and calculate comprehensive metrics"""
        neural_agents = [agent for agent in env.agents if isinstance(agent, NeuralAgent)]
        
        if not neural_agents:
            print("⚠️  No neural agents survived this generation!")
            return
        
        # Separate by species
        herbivores = [a for a in neural_agents if a.species_type == SpeciesType.HERBIVORE]
        carnivores = [a for a in neural_agents if a.species_type == SpeciesType.CARNIVORE]
        
        # Calculate comprehensive metrics
        metrics = GenerationMetrics(
            generation=self.current_generation,
            step_range=(self.generation_start_step, end_step),
            
            # Population
            herbivore_count=len(herbivores),
            carnivore_count=len(carnivores),
            
            # Fitness metrics
            avg_herbivore_fitness=np.mean([a.brain.fitness_score for a in herbivores]) if herbivores else 0,
            avg_carnivore_fitness=np.mean([a.brain.fitness_score for a in carnivores]) if carnivores else 0,
            max_herbivore_fitness=np.max([a.brain.fitness_score for a in herbivores]) if herbivores else 0,
            max_carnivore_fitness=np.max([a.brain.fitness_score for a in carnivores]) if carnivores else 0,
            min_herbivore_fitness=np.min([a.brain.fitness_score for a in herbivores]) if herbivores else 0,
            min_carnivore_fitness=np.min([a.brain.fitness_score for a in carnivores]) if carnivores else 0,
            
            # Behavioral metrics
            avg_decisions_per_agent=np.mean([a.brain.decisions_made for a in neural_agents]),
            total_reproductions=sum(a.offspring_count for a in neural_agents),
            total_hunts=sum(a.lifetime_successful_hunts for a in neural_agents),
            total_food_consumed=sum(a.lifetime_food_consumed for a in neural_agents),
            
            # Neural network analysis
            avg_weight_magnitude=self._calculate_avg_weight_magnitude(neural_agents),
            weight_diversity=self._calculate_weight_diversity(neural_agents),
            
            # Survival analysis
            avg_lifespan=np.mean([a.survival_time for a in neural_agents]),
            extinction_events=int(len(herbivores) == 0) + int(len(carnivores) == 0)
        )
        
        self.generation_metrics.append(metrics)
        
        # Print generation summary
        self._print_generation_summary(metrics)
        
        # Set baseline if first generation
        if self.initial_baseline is None:
            self.initial_baseline = metrics
    
    def _calculate_avg_weight_magnitude(self, agents: List[NeuralAgent]) -> float:
        """Calculate average magnitude of neural network weights"""
        total_magnitude = 0
        total_weights = 0
        
        for agent in agents:
            # Get all weights from neural network
            weights = []
            if hasattr(agent.brain, 'weights_input_hidden'):
                weights.extend(agent.brain.weights_input_hidden.flatten())
            if hasattr(agent.brain, 'weights_hidden_output'):
                weights.extend(agent.brain.weights_hidden_output.flatten())
            
            if weights:
                total_magnitude += np.mean(np.abs(weights))
                total_weights += 1
        
        return total_magnitude / total_weights if total_weights > 0 else 0
    
    def _calculate_weight_diversity(self, agents: List[NeuralAgent]) -> float:
        """Calculate diversity (standard deviation) of weights across population"""
        all_weights = []
        
        for agent in agents:
            weights = []
            if hasattr(agent.brain, 'weights_input_hidden'):
                weights.extend(agent.brain.weights_input_hidden.flatten())
            if hasattr(agent.brain, 'weights_hidden_output'):
                weights.extend(agent.brain.weights_hidden_output.flatten())
            
            if weights:
                all_weights.extend(weights)
        
        return np.std(all_weights) if all_weights else 0
    
    def _print_generation_summary(self, metrics: GenerationMetrics):
        """Print comprehensive generation summary"""
        print(f"\n📊 Generation {metrics.generation} Summary (Steps {metrics.step_range[0]}-{metrics.step_range[1]})")
        print("=" * 70)
        
        # Population
        print(f"🦌 Herbivores: {metrics.herbivore_count} | 🐺 Carnivores: {metrics.carnivore_count}")
        
        # Fitness evolution
        if self.initial_baseline and metrics.generation > 1:
            herb_improvement = ((metrics.avg_herbivore_fitness / self.initial_baseline.avg_herbivore_fitness) - 1) * 100 if self.initial_baseline.avg_herbivore_fitness > 0 else 0
            carn_improvement = ((metrics.avg_carnivore_fitness / self.initial_baseline.avg_carnivore_fitness) - 1) * 100 if self.initial_baseline.avg_carnivore_fitness > 0 else 0
            
            print(f"📈 Fitness Evolution:")
            print(f"   Herbivores: {metrics.avg_herbivore_fitness:.1f} ({herb_improvement:+.1f}% from baseline)")
            print(f"   Carnivores: {metrics.avg_carnivore_fitness:.1f} ({carn_improvement:+.1f}% from baseline)")
        else:
            print(f"📈 Average Fitness: H={metrics.avg_herbivore_fitness:.1f}, C={metrics.avg_carnivore_fitness:.1f}")
        
        # Top performers
        print(f"🏆 Peak Fitness: H={metrics.max_herbivore_fitness:.1f}, C={metrics.max_carnivore_fitness:.1f}")
        
        # Behavioral metrics
        print(f"🧠 Neural Activity: {metrics.avg_decisions_per_agent:.0f} decisions/agent")
        print(f"🍼 Reproductions: {metrics.total_reproductions} | 🎯 Hunts: {metrics.total_hunts}")
        
        # Neural evolution indicators
        print(f"⚙️  Weight Magnitude: {metrics.avg_weight_magnitude:.3f} | Diversity: {metrics.weight_diversity:.3f}")
        print(f"⏱️  Average Lifespan: {metrics.avg_lifespan:.0f} steps")
        
        if metrics.extinction_events > 0:
            print("💀 Extinction events detected!")
    
    def create_evolution_dashboard(self, save_path: str = "neural_evolution_dashboard.png"):
        """Create comprehensive evolution dashboard"""
        if len(self.generation_metrics) < 2:
            print("⚠️  Need at least 2 generations to create evolution dashboard")
            return
        
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('🧬 Neural Network Evolution Dashboard', fontsize=20, fontweight='bold')
        
        generations = [m.generation for m in self.generation_metrics]
        
        # 1. Fitness Evolution Over Generations
        ax1.set_title('📈 Fitness Evolution Across Generations', fontsize=14, fontweight='bold')
        herb_avg_fitness = [m.avg_herbivore_fitness for m in self.generation_metrics]
        carn_avg_fitness = [m.avg_carnivore_fitness for m in self.generation_metrics]
        herb_max_fitness = [m.max_herbivore_fitness for m in self.generation_metrics]
        carn_max_fitness = [m.max_carnivore_fitness for m in self.generation_metrics]
        
        ax1.plot(generations, herb_avg_fitness, 'g-', linewidth=3, label='Herbivore Avg', marker='o')
        ax1.plot(generations, carn_avg_fitness, 'r-', linewidth=3, label='Carnivore Avg', marker='s')
        ax1.plot(generations, herb_max_fitness, 'g--', linewidth=2, label='Herbivore Peak', alpha=0.7)
        ax1.plot(generations, carn_max_fitness, 'r--', linewidth=2, label='Carnivore Peak', alpha=0.7)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Population Dynamics
        ax2.set_title('🦌🐺 Population Evolution', fontsize=14, fontweight='bold')
        herb_counts = [m.herbivore_count for m in self.generation_metrics]
        carn_counts = [m.carnivore_count for m in self.generation_metrics]
        
        ax2.bar([g-0.2 for g in generations], herb_counts, 0.4, label='Herbivores', color='green', alpha=0.7)
        ax2.bar([g+0.2 for g in generations], carn_counts, 0.4, label='Carnivores', color='red', alpha=0.7)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Population Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Neural Network Complexity Evolution
        ax3.set_title('⚙️ Neural Network Weight Evolution', fontsize=14, fontweight='bold')
        weight_magnitudes = [m.avg_weight_magnitude for m in self.generation_metrics]
        weight_diversities = [m.weight_diversity for m in self.generation_metrics]
        
        ax3_twin = ax3.twinx()
        ax3.plot(generations, weight_magnitudes, 'b-', linewidth=3, label='Weight Magnitude', marker='D')
        ax3_twin.plot(generations, weight_diversities, 'purple', linewidth=3, label='Weight Diversity', marker='x', linestyle='--')
        
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Average Weight Magnitude', color='blue')
        ax3_twin.set_ylabel('Weight Diversity (Std Dev)', color='purple')
        ax3.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 4. Behavioral Metrics
        ax4.set_title('🧠 Behavioral Evolution', fontsize=14, fontweight='bold')
        decisions = [m.avg_decisions_per_agent for m in self.generation_metrics]
        reproductions = [m.total_reproductions for m in self.generation_metrics]
        hunts = [m.total_hunts for m in self.generation_metrics]
        
        ax4_twin = ax4.twinx()
        ax4.plot(generations, decisions, 'orange', linewidth=3, label='Decisions/Agent', marker='o')
        ax4_twin.bar([g-0.15 for g in generations], reproductions, 0.3, label='Reproductions', color='pink', alpha=0.7)
        ax4_twin.bar([g+0.15 for g in generations], hunts, 0.3, label='Hunts', color='brown', alpha=0.7)
        
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Avg Decisions per Agent', color='orange')
        ax4_twin.set_ylabel('Total Events')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 5. Survival Analysis
        ax5.set_title('⏱️ Survival & Lifespan Evolution', fontsize=14, fontweight='bold')
        lifespans = [m.avg_lifespan for m in self.generation_metrics]
        extinction_events = [m.extinction_events for m in self.generation_metrics]
        
        ax5.plot(generations, lifespans, 'teal', linewidth=4, label='Average Lifespan', marker='s')
        ax5.scatter([g for g, e in zip(generations, extinction_events) if e > 0], 
                   [l for l, e in zip(lifespans, extinction_events) if e > 0], 
                   c='red', s=200, marker='X', label='Extinction Events', alpha=0.8)
        
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Average Lifespan (steps)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Improvement Metrics
        ax6.set_title('📊 Learning Progress Summary', fontsize=14, fontweight='bold')
        
        if len(self.generation_metrics) >= 3:
            # Calculate learning trends
            recent_gens = self.generation_metrics[-3:]  # Last 3 generations
            
            metrics_data = {
                'Herbivore\nFitness': np.mean([g.avg_herbivore_fitness for g in recent_gens]),
                'Carnivore\nFitness': np.mean([g.avg_carnivore_fitness for g in recent_gens]),
                'Population\nStability': 1.0 - (np.std([g.herbivore_count + g.carnivore_count for g in recent_gens]) / 20),
                'Neural\nComplexity': np.mean([g.avg_weight_magnitude for g in recent_gens]) * 10,
                'Behavioral\nRichness': np.mean([g.avg_decisions_per_agent for g in recent_gens]) / 100
            }
            
            bars = ax6.bar(metrics_data.keys(), metrics_data.values(), 
                          color=['green', 'red', 'blue', 'purple', 'orange'], alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_data.values()):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax6.set_ylabel('Normalized Score (0-1 scale)')
        ax6.set_title('🎯 Current Performance Metrics (Last 3 Generations)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Evolution dashboard saved to: {save_path}")
        
        return save_path
    
    def save_evolution_data(self, filepath: str = "neural_evolution_data.json"):
        """Save evolution tracking data to JSON"""
        data = {
            'generation_metrics': [asdict(m) for m in self.generation_metrics],
            'fitness_history': self.fitness_history,
            'current_generation': self.current_generation,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Evolution data saved to: {filepath}")
    
    def detect_learning_patterns(self):
        """Analyze learning patterns and provide insights"""
        if len(self.generation_metrics) < 3:
            return "Need at least 3 generations to detect learning patterns"
        
        insights = []
        recent_gens = self.generation_metrics[-3:]
        
        # Fitness trend analysis
        herb_trend = np.polyfit(range(len(recent_gens)), [g.avg_herbivore_fitness for g in recent_gens], 1)[0]
        carn_trend = np.polyfit(range(len(recent_gens)), [g.avg_carnivore_fitness for g in recent_gens], 1)[0]
        
        if herb_trend > 0.5:
            insights.append("🦌 Herbivores showing strong learning trend!")
        elif herb_trend < -0.5:
            insights.append("🦌 Herbivore performance declining - may need parameter adjustment")
        
        if carn_trend > 0.5:
            insights.append("🐺 Carnivores rapidly improving their hunting strategies!")
        elif carn_trend < -0.5:
            insights.append("🐺 Carnivore fitness declining - possibly overfitting")
        
        # Population stability
        pop_variance = np.var([g.herbivore_count + g.carnivore_count for g in recent_gens])
        if pop_variance < 5:
            insights.append("⚖️ Population reaching stable equilibrium")
        elif pop_variance > 20:
            insights.append("⚠️ Population showing high volatility")
        
        # Neural complexity
        weight_trend = np.polyfit(range(len(recent_gens)), [g.avg_weight_magnitude for g in recent_gens], 1)[0]
        if weight_trend > 0.01:
            insights.append("🧠 Neural networks becoming more complex (good learning sign)")
        elif weight_trend < -0.01:
            insights.append("🧠 Networks simplifying - possibly converging on optimal strategy")
        
        return insights

if __name__ == "__main__":
    print("🧬 Neural Evolution Tracker - Test Mode")
    print("=" * 50)
    
    # Create a mock tracker with sample data for testing
    tracker = NeuralEvolutionTracker()
    
    # Add some sample generation data
    import random
    for gen in range(1, 6):
        metrics = GenerationMetrics(
            generation=gen,
            step_range=(gen*1000, (gen+1)*1000),
            herbivore_count=15 + random.randint(-3, 5),
            carnivore_count=6 + random.randint(-2, 3),
            avg_herbivore_fitness=10 + gen * 2 + random.uniform(-1, 1),
            avg_carnivore_fitness=8 + gen * 1.5 + random.uniform(-1, 1),
            max_herbivore_fitness=15 + gen * 3,
            max_carnivore_fitness=12 + gen * 2.5,
            min_herbivore_fitness=5 + gen * 0.5,
            min_carnivore_fitness=3 + gen * 0.3,
            avg_decisions_per_agent=50 + gen * 10,
            total_reproductions=8 + gen * 2,
            total_hunts=3 + gen,
            total_food_consumed=25 + gen * 5,
            avg_weight_magnitude=0.3 + gen * 0.02,
            weight_diversity=0.8 + gen * 0.1,
            avg_lifespan=200 + gen * 50,
            extinction_events=0
        )
        tracker.generation_metrics.append(metrics)
    
    # Test dashboard creation
    tracker.create_evolution_dashboard("test_evolution_dashboard.png")
    
    # Test insights
    insights = tracker.detect_learning_patterns()
    print("\n🔍 Learning Pattern Insights:")
    for insight in insights:
        print(f"  {insight}")
    
    print("\n✅ Neural Evolution Tracker tested successfully!")
