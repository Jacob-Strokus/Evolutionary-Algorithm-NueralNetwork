"""
Genetic algorithm for inter-generational evolution.

Usage
-----
At the end of each generation (every N steps), call:

    ga = GeneticAlgorithm(config)
    next_gen = ga.evolve(current_agents, env)
    env.agents = next_gen

The algorithm:
  1. Separates agents by species.
  2. Preserves the top *elite_fraction* of each species unchanged.
  3. Fills the remaining target population via tournament selection + crossover.
  4. Applies mutation to every non-elite offspring.
  5. If a species has fewer survivors than its minimum viable count, randomly
     spawns additional agents bred from whatever survivors exist.
"""

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.core.ecosystem import Environment, Position, SpeciesType
from src.neural.neural_network import NeuralNetwork, NetworkConfig
from src.neural.neural_agents import NeuralAgent


@dataclass
class GAConfig:
    # Target population sizes
    target_herbivores : int   = 20
    target_carnivores : int   = 8

    # Minimum viable populations (below this, inject new individuals)
    min_herbivores    : int   = 6
    min_carnivores    : int   = 3

    # Selection pressure
    elite_fraction    : float = 0.20   # top fraction preserved unchanged
    tournament_size   : int   = 4      # k-way tournament for parent selection
    crossover_rate    : float = 0.75   # probability of sexual vs. asexual reproduction

    # Mutation (applied to every non-elite offspring)
    mutation_rate     : float = 0.15
    mutation_strength : float = 0.20


class GeneticAlgorithm:
    """Produces the next generation of agents from the survivors of the current one."""

    def __init__(self, config: GAConfig = None, net_config: NetworkConfig = None):
        self.config     = config     or GAConfig()
        self.net_config = net_config or NetworkConfig()
        self.generation = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def evolve(self, agents: List[NeuralAgent], env: Environment) -> List[NeuralAgent]:
        """
        Build and return the next generation.
        Agents are placed at random positions; their ages and energy are reset.
        """
        self.generation += 1
        next_gen: List[NeuralAgent] = []

        targets = {
            SpeciesType.HERBIVORE: self.config.target_herbivores,
            SpeciesType.CARNIVORE: self.config.target_carnivores,
        }

        for species in (SpeciesType.HERBIVORE, SpeciesType.CARNIVORE):
            pool = [a for a in agents if a.species_type == species]
            target = targets[species]
            next_gen.extend(self._evolve_species(pool, species, target, env))

        return next_gen

    # ------------------------------------------------------------------ #
    # Per-species evolution                                                #
    # ------------------------------------------------------------------ #

    def _evolve_species(
        self,
        pool   : List[NeuralAgent],
        species: SpeciesType,
        target : int,
        env    : Environment,
    ) -> List[NeuralAgent]:

        if not pool:
            # Species went fully extinct — seed from scratch
            return self._seed_random(species, target, env)

        # Sort by fitness (descending)
        pool.sort(key=lambda a: a.brain.fitness, reverse=True)

        elite_count = max(1, round(len(pool) * self.config.elite_fraction))
        elites      = pool[:elite_count]
        offspring   = []

        # Carry elites over with reset state but preserved weights
        for elite in elites:
            child = self._reset_agent(elite, env)
            offspring.append(child)

        # Fill up to target count by breeding
        while len(offspring) < target:
            parent1 = self._tournament(pool)
            if len(pool) >= 2 and random.random() < self.config.crossover_rate:
                parent2 = self._tournament(pool, exclude=parent1)
                brain   = parent1.brain.crossover(parent2.brain)
            else:
                brain = parent1.brain.copy()

            brain.mutate()
            brain.fitness = 0.0
            brain.reset_state()

            child = self._make_agent(species, brain, env)
            offspring.append(child)

        return offspring

    # ------------------------------------------------------------------ #
    # Selection                                                            #
    # ------------------------------------------------------------------ #

    def _tournament(
        self,
        pool   : List[NeuralAgent],
        exclude: Optional[NeuralAgent] = None,
    ) -> NeuralAgent:
        """k-way tournament selection; higher fitness wins."""
        candidates = [a for a in pool if a is not exclude]
        if not candidates:
            candidates = pool  # fallback if exclude == the whole pool

        k         = min(self.config.tournament_size, len(candidates))
        entrants  = random.sample(candidates, k)
        return max(entrants, key=lambda a: a.brain.fitness)

    # ------------------------------------------------------------------ #
    # Agent construction helpers                                           #
    # ------------------------------------------------------------------ #

    def _make_agent(
        self,
        species: SpeciesType,
        brain  : NeuralNetwork,
        env    : Environment,
    ) -> NeuralAgent:
        """Create a fresh agent with a given brain at a random position."""
        pos = Position(
            random.uniform(10.0, env.width  - 10.0),
            random.uniform(10.0, env.height - 10.0),
        )
        agent = NeuralAgent(
            species_type = species,
            position     = pos,
            agent_id     = env.next_agent_id(),
            brain        = brain,
            config       = self.net_config,
        )
        return agent

    def _reset_agent(self, source: NeuralAgent, env: Environment) -> NeuralAgent:
        """
        Clone an elite agent: same brain weights, but new position,
        reset age/energy/fitness, and cleared recurrent state.
        """
        brain = source.brain.copy()
        brain.fitness = 0.0
        brain.reset_state()
        return self._make_agent(source.species_type, brain, env)

    def _seed_random(
        self,
        species: SpeciesType,
        count  : int,
        env    : Environment,
    ) -> List[NeuralAgent]:
        """Create *count* brand-new random agents when a species goes extinct."""
        agents = []
        for _ in range(count):
            brain = NeuralNetwork(self.net_config)
            agent = self._make_agent(species, brain, env)
            agents.append(agent)
        return agents

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def get_stats(self, agents: List[NeuralAgent]) -> dict:
        herbs = [a for a in agents if a.species_type == SpeciesType.HERBIVORE]
        carns = [a for a in agents if a.species_type == SpeciesType.CARNIVORE]

        def species_stats(group):
            if not group:
                return {"count": 0, "avg_fitness": 0.0, "max_fitness": 0.0}
            fitnesses = [a.brain.fitness for a in group]
            return {
                "count"      : len(group),
                "avg_fitness": round(float(np.mean(fitnesses)), 2),
                "max_fitness": round(float(np.max(fitnesses)),  2),
            }

        return {
            "generation" : self.generation,
            "herbivores" : species_stats(herbs),
            "carnivores" : species_stats(carns),
        }
