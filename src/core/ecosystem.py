"""
Ecosystem - core simulation primitives.

Defines the world (Environment), food sources, positions, and species types.
The Environment runs one clean step loop: move → feed → hunt → reproduce → cleanup.
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class SpeciesType(Enum):
    HERBIVORE = "herbivore"
    CARNIVORE  = "carnivore"


@dataclass
class Position:
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Food:
    """A food source that respawns after a fixed number of steps."""

    def __init__(self, position: Position, energy_value: float = 30.0, regen_time: int = 80):
        self.position     = position
        self.energy_value = energy_value
        self.regen_time   = regen_time
        self.is_available = True
        self._regen_counter = 0

    def consume(self) -> float:
        """Consume this food source and return the energy gained."""
        self.is_available   = False
        self._regen_counter = 0
        return self.energy_value

    def tick(self):
        """Advance the regeneration timer by one simulation step."""
        if not self.is_available:
            self._regen_counter += 1
            if self._regen_counter >= self.regen_time:
                self.is_available   = True
                self._regen_counter = 0


class Environment:
    """
    The simulation world.

    Runs a four-phase step:
      1. Neural decisions + movement   (each agent calls think_and_move)
      2. Feeding interactions           (herbivores graze, carnivores hunt)
      3. Reproduction                   (agents that want to and can, do)
      4. Cleanup                        (remove dead, add offspring, regen food)
    """

    FOOD_COUNT  = 80
    FOOD_ENERGY = 30.0
    FOOD_REGEN  = 60   # steps until a consumed food source respawns

    def __init__(self, width: int = 100, height: int = 100):
        self.width      = width
        self.height     = height
        self.time_step  = 0
        self._next_id   = 1
        self.agents: List   = []
        self.food_sources: List[Food] = []
        self._spawn_food(self.FOOD_COUNT)

    # ------------------------------------------------------------------ #
    # Agent / food management                                              #
    # ------------------------------------------------------------------ #

    def next_agent_id(self) -> int:
        aid = self._next_id
        self._next_id += 1
        return aid

    def add_agents(self, agents):
        self.agents.extend(agents)

    def clamp(self, agent):
        """Keep an agent strictly inside the world boundaries."""
        agent.position.x = max(0.0, min(float(self.width),  agent.position.x))
        agent.position.y = max(0.0, min(float(self.height), agent.position.y))

    def _spawn_food(self, n: int):
        for _ in range(n):
            pos = Position(
                random.uniform(5.0, self.width  - 5.0),
                random.uniform(5.0, self.height - 5.0),
            )
            self.food_sources.append(Food(pos, self.FOOD_ENERGY, self.FOOD_REGEN))

    # ------------------------------------------------------------------ #
    # Nearest-entity helpers used by the sensor system                     #
    # ------------------------------------------------------------------ #

    def nearest_food(self, position: Position, vision_range: float, n: int = 3):
        """
        Return up to *n* available food sources within *vision_range*,
        as a list of (distance, Food) tuples sorted nearest-first.
        """
        visible = []
        for food in self.food_sources:
            if food.is_available:
                d = position.distance_to(food.position)
                if d <= vision_range:
                    visible.append((d, food))
        visible.sort(key=lambda t: t[0])
        return visible[:n]

    def nearest_agents(
        self,
        position: Position,
        vision_range: float,
        species: SpeciesType,
        exclude=None,
        n: int = 3,
    ):
        """
        Return up to *n* alive agents of the given species within
        *vision_range*, as (distance, agent) tuples sorted nearest-first.
        """
        visible = []
        for agent in self.agents:
            if agent is exclude or not agent.is_alive:
                continue
            if agent.species_type == species:
                d = position.distance_to(agent.position)
                if d <= vision_range:
                    visible.append((d, agent))
        visible.sort(key=lambda t: t[0])
        return visible[:n]

    # ------------------------------------------------------------------ #
    # Main step                                                            #
    # ------------------------------------------------------------------ #

    def step(self):
        alive = [a for a in self.agents if a.is_alive]

        # Phase 1 – Neural decisions + movement
        for agent in alive:
            agent.think_and_move(self)

        # Phase 2 – Feeding
        # Reset hunt-lock flags so each prey can only be killed once per step.
        for agent in alive:
            agent._targeted = False

        herbivores = [a for a in alive if a.species_type == SpeciesType.HERBIVORE]
        carnivores = [a for a in alive if a.species_type == SpeciesType.CARNIVORE]

        for herb in herbivores:
            self._graze(herb)

        for carn in carnivores:
            self._hunt(carn, herbivores)

        # Phase 3 – Reproduction
        offspring = []
        for agent in alive:
            if agent._wants_to_reproduce and agent.can_reproduce():
                child = agent.reproduce(self)
                if child is not None:
                    offspring.append(child)

        # Phase 4 – Cleanup
        self.agents = [a for a in self.agents if a.is_alive]
        self.agents.extend(offspring)

        for food in self.food_sources:
            food.tick()

        self.time_step += 1

    # ------------------------------------------------------------------ #
    # Feeding helpers                                                      #
    # ------------------------------------------------------------------ #

    def _graze(self, herbivore):
        """Let a herbivore eat the nearest available food within eat_range."""
        for food in self.food_sources:
            if food.is_available:
                if herbivore.position.distance_to(food.position) <= herbivore.eat_range:
                    herbivore.eat(food.consume())
                    break  # one source per step

    def _hunt(self, carnivore, herbivores: list):
        """Let a carnivore attempt to kill the nearest untargeted prey."""
        best      = None
        best_dist = float("inf")

        for prey in herbivores:
            if not prey.is_alive or prey._targeted:
                continue
            d = carnivore.position.distance_to(prey.position)
            if d <= carnivore.hunt_range and d < best_dist:
                best      = prey
                best_dist = d

        if best is None:
            return

        # Hunt success chance: increases with carnivore energy and decreases with distance.
        energy_factor = min(0.55, (carnivore.energy / carnivore.max_energy) * 0.7)
        dist_factor   = max(0.0,  1.0 - best_dist / carnivore.hunt_range) * 0.3
        chance        = min(0.80, 0.30 + energy_factor + dist_factor)

        if random.random() < chance:
            best._targeted = True
            best.is_alive  = False
            carnivore.eat(best.energy * 0.7)
            carnivore.prey_captures += 1

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        herbs = [a for a in self.agents if a.species_type == SpeciesType.HERBIVORE]
        carns = [a for a in self.agents if a.species_type == SpeciesType.CARNIVORE]
        avail = sum(1 for f in self.food_sources if f.is_available)

        return {
            "time_step"           : self.time_step,
            "herbivores"          : len(herbs),
            "carnivores"          : len(carns),
            "total_agents"        : len(self.agents),
            "available_food"      : avail,
            "avg_herbivore_energy": sum(a.energy for a in herbs) / len(herbs) if herbs else 0.0,
            "avg_carnivore_energy": sum(a.energy for a in carns) / len(carns) if carns else 0.0,
        }
