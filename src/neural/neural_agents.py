"""
Neural agents — the simulation's core actors.

Each NeuralAgent owns one NeuralNetwork (its "brain") and runs the full
sense → decide → act loop every step.  Fitness is tracked cumulatively and
used by the genetic algorithm for selection.

SensorSystem is a pure-function helper that builds the 24-element input
vector the network expects and decodes the 4-element output vector into
named action components.

Sensor layout (24 inputs):
  [0]     energy level              (0–1)
  [1]     age                       (0–1, 1 = 1 500 steps)
  [2]     reproduction readiness    (0/1)
  [3-8]   3 × (food distance, food angle)
  [9-14]  3 × (threat/prey distance, threat/prey angle)
  [15]    x-axis boundary proximity (0 = at wall, 1 = at centre)
  [16]    y-axis boundary proximity (0 = at wall, 1 = at centre)
  [17]    time since last food      (0–1, 1 = 100+ steps)
  [18-23] 3 × (same-species distance, same-species angle)
            — carnivores: where are my pack members?
            — herbivores: where is my herd?

Action layout (4 outputs, all sigmoid → [0, 1]):
  [0]  move_x   → remapped to [-1, 1]
  [1]  move_y   → remapped to [-1, 1]
  [2]  speed    → kept as [0, 1], scales with species max speed
  [3]  reproduce → threshold at 0.5
"""

import math
import random
from typing import Optional

import numpy as np

from src.core.ecosystem import Position, SpeciesType
from src.neural.neural_network import NeuralNetwork, NetworkConfig


# ======================================================================= #
# Species-specific constants                                               #
# ======================================================================= #

_HERBIVORE = {
    "starting_energy"  : 100.0,
    "max_energy"       : 150.0,
    "energy_decay"     : 0.30,   # energy lost per step (idle)
    "speed"            : 2.0,    # world-units per step at full speed
    "vision_range"     : 20.0,
    "eat_range"        : 2.5,    # must be within this distance to graze
    "repro_threshold"  : 80.0,   # minimum energy to attempt reproduction (was 100)
    "repro_cost"       : 35.0,   # energy deducted from parent on reproduction (was 50)
    "repro_cooldown"   : 120,    # steps before parent can reproduce again (was 150)
    "min_repro_age"    : 40,     # must be at least this old
    "max_age"          : 2000,
}

_CARNIVORE = {
    "starting_energy"     : 120.0,
    "max_energy"          : 200.0,
    "energy_decay"        : 0.35,   # base idle decay (was 0.40)
    "starvation_after"    : 100,    # steps without food before extra decay kicks in (was 50)
    "starvation_decay"    : 0.65,   # extra energy lost per step while starving (total 1.0/step; was 1.20 = 1.60/step)
    "speed"               : 2.6,
    "vision_range"        : 25.0,
    "hunt_range"          : 5.0,    # catch radius in world units (was 4.5)
    "repro_threshold"     : 120.0,  # one kill puts them above this (was 140)
    "repro_cost"          : 45.0,   # energy deducted from parent on reproduction (was 60)
    "repro_cooldown"      : 160,    # steps before parent can reproduce again (was 200)
    "min_repro_age"       : 60,
    "max_age"             : 2000,
}

_PARAMS = {
    SpeciesType.HERBIVORE: _HERBIVORE,
    SpeciesType.CARNIVORE: _CARNIVORE,
}


# ======================================================================= #
# Sensor system                                                            #
# ======================================================================= #

class SensorSystem:
    """Builds the network's input vector from an agent's current perception."""

    N_INPUTS = 24

    @staticmethod
    def sense(agent: "NeuralAgent", env) -> np.ndarray:
        p      = _PARAMS[agent.species_type]
        inputs = np.zeros(SensorSystem.N_INPUTS)

        # Internal state
        inputs[0] = min(1.0, agent.energy / agent.max_energy)
        inputs[1] = min(1.0, agent.age / 1500.0)
        inputs[2] = 1.0 if agent.can_reproduce() else 0.0

        # Food sensing (up to 3 sources)
        food_hits = env.nearest_food(agent.position, agent.vision_range, n=3)
        for i, (dist, food) in enumerate(food_hits):
            base         = 3 + i * 2
            inputs[base]     = dist / agent.vision_range          # 0=here, 1=edge
            inputs[base + 1] = SensorSystem._angle(agent.position, food.position)
        # unfilled slots stay at 0 (no food in that slot)

        # Threat / prey sensing (up to 3 targets)
        if agent.species_type == SpeciesType.HERBIVORE:
            target_species = SpeciesType.CARNIVORE   # herbivores fear carnivores
        else:
            target_species = SpeciesType.HERBIVORE   # carnivores hunt herbivores

        target_hits = env.nearest_agents(agent.position, agent.vision_range,
                                         target_species, exclude=agent, n=3)
        for i, (dist, target) in enumerate(target_hits):
            base         = 9 + i * 2
            inputs[base]     = dist / agent.vision_range
            inputs[base + 1] = SensorSystem._angle(agent.position, target.position)

        # Boundary proximity (0 = touching wall, 1 = at centre)
        inputs[15] = min(agent.position.x, env.width  - agent.position.x) / (env.width  / 2.0)
        inputs[16] = min(agent.position.y, env.height - agent.position.y) / (env.height / 2.0)

        # Time since last food (hunger signal)
        inputs[17] = min(1.0, agent._steps_since_food / 100.0)

        # Same-species positions (up to 3 nearest companions)
        # Carnivores: pack awareness for coordinated hunting
        # Herbivores: herd awareness for group movement / safety in numbers
        same_hits = env.nearest_agents(agent.position, agent.vision_range,
                                        agent.species_type, exclude=agent, n=3)
        for i, (dist, companion) in enumerate(same_hits):
            base             = 18 + i * 2
            inputs[base]     = dist / agent.vision_range
            inputs[base + 1] = SensorSystem._angle(agent.position, companion.position)
        # unfilled slots stay 0 (no companion in that slot)

        return inputs

    @staticmethod
    def decode(outputs: np.ndarray) -> dict:
        """Convert the 4 sigmoid outputs into named action components."""
        return {
            "move_x"   : (outputs[0] - 0.5) * 2.0,  # [-1, 1]
            "move_y"   : (outputs[1] - 0.5) * 2.0,  # [-1, 1]
            "speed"    : float(outputs[2]),           # [0, 1]
            "reproduce": bool(outputs[3] > 0.5),
        }

    @staticmethod
    def _angle(src: Position, dst: Position) -> float:
        """Signed angle from src to dst, normalised to [-1, 1]."""
        return math.atan2(dst.y - src.y, dst.x - src.x) / math.pi


# ======================================================================= #
# Neural agent                                                             #
# ======================================================================= #

class NeuralAgent:
    """
    A simulation agent driven by a NeuralNetwork.

    The agent itself has no behaviour logic beyond running its neural network
    and maintaining its physical/biological state.  All world interactions
    (eating, hunting, reproduction) are initiated by the Environment, which
    calls helpers on the agent (eat, reproduce) after confirming proximity.
    """

    def __init__(
        self,
        species_type: SpeciesType,
        position    : Position,
        agent_id    : int,
        brain       : Optional[NeuralNetwork] = None,
        config      : Optional[NetworkConfig] = None,
    ):
        self.species_type = species_type
        self.position     = position
        self.agent_id     = agent_id

        p = _PARAMS[species_type]

        # Biological state
        self.energy       = p["starting_energy"]
        self.max_energy   = p["max_energy"]
        self.age          = 0
        self.is_alive     = True

        # Species attributes (some are read directly by Environment)
        self.vision_range = p["vision_range"]
        self.speed        = p["speed"]
        self.eat_range    = p.get("eat_range",   0.0)
        self.hunt_range   = p.get("hunt_range",  0.0)

        # Reproduction state
        self._repro_threshold = p["repro_threshold"]
        self._repro_cost      = p["repro_cost"]
        self._repro_cooldown  = p["repro_cooldown"]
        self._min_repro_age   = p["min_repro_age"]
        self._repro_timer     = 0

        # Step-private flags (set and read within the same Environment.step call)
        self._wants_to_reproduce = False
        self._targeted           = False   # True once a carnivore wins a hunt roll on this prey

        # Starvation tracking (carnivores only)
        self._steps_since_food = 0

        # Fitness bookkeeping
        self.food_consumed  = 0.0   # total energy received from food/prey
        self.prey_captures  = 0
        self.offspring_born = 0

        # Brain
        cfg        = config or NetworkConfig()
        self.brain = brain  if brain  is not None else NeuralNetwork(cfg)
        self._cfg  = cfg

    # ------------------------------------------------------------------ #
    # Core step                                                            #
    # ------------------------------------------------------------------ #

    def think_and_move(self, env):
        """Run the neural network and apply the resulting movement."""
        inputs  = SensorSystem.sense(self, env)
        outputs = self.brain.forward(inputs)
        self._last_inputs  = inputs   # stored for the web inspector
        self._last_outputs = outputs
        actions = SensorSystem.decode(outputs)

        self._wants_to_reproduce = actions["reproduce"]

        # Move
        mx   = actions["move_x"]
        my   = actions["move_y"]
        spd  = actions["speed"] * self.speed

        # Normalise direction vector so diagonal moves aren't faster
        mag = math.sqrt(mx * mx + my * my)
        if mag > 1e-6:
            mx /= mag
            my /= mag

        self.position.x += mx * spd
        self.position.y += my * spd
        env.clamp(self)

        # Energy and age
        self._apply_energy_decay()
        self.age += 1

        if self.energy <= 0 or self.age >= _PARAMS[self.species_type]["max_age"]:
            self.is_alive = False

        # Small survival bonus each step — directly rewards longevity.
        # Herbivores that evade predators and carnivores that hunt effectively
        # both live longer, accumulating this reward over food/kill rewards.
        self.brain.fitness += 0.05

    def _apply_energy_decay(self):
        p = _PARAMS[self.species_type]
        cost = p["energy_decay"]

        if self.species_type == SpeciesType.CARNIVORE:
            self._steps_since_food += 1
            if self._steps_since_food > p["starvation_after"]:
                cost += p["starvation_decay"]

        self.energy -= cost
        if self.energy < 0:
            self.energy = 0.0

    # ------------------------------------------------------------------ #
    # Interaction hooks (called by Environment)                            #
    # ------------------------------------------------------------------ #

    def eat(self, energy: float):
        """Receive energy from a food source or a successful hunt."""
        self.energy        = min(self.max_energy, self.energy + energy)
        self.food_consumed += energy
        self._steps_since_food = 0

        # Update fitness: reward efficiency over raw accumulation
        self.brain.fitness += energy * 0.5

    def can_reproduce(self) -> bool:
        return (
            self.is_alive
            and self.energy    >= self._repro_threshold
            and self.age       >= self._min_repro_age
            and self._repro_timer <= 0
        )

    def reproduce(self, env) -> Optional["NeuralAgent"]:
        """
        Create an asexual offspring with a mutated copy of this brain.
        The parent pays the reproduction cost and starts a cooldown.
        """
        if not self.can_reproduce():
            return None

        self.energy      -= self._repro_cost
        self._repro_timer = self._repro_cooldown
        self.offspring_born += 1
        self.brain.fitness  += 30.0   # reward for reproducing

        offset   = 6.0
        child_pos = Position(
            self.position.x + random.uniform(-offset, offset),
            self.position.y + random.uniform(-offset, offset),
        )
        child_pos.x = max(0.0, min(float(env.width),  child_pos.x))
        child_pos.y = max(0.0, min(float(env.height), child_pos.y))

        child_brain = self.brain.copy()
        child_brain.mutate()
        child_brain.fitness = 0.0  # fresh fitness slate
        child_brain.reset_state()

        return NeuralAgent(
            species_type = self.species_type,
            position     = child_pos,
            agent_id     = env.next_agent_id(),
            brain        = child_brain,
            config       = self._cfg,
        )

    # ------------------------------------------------------------------ #
    # Per-step bookkeeping                                                 #
    # ------------------------------------------------------------------ #

    def tick_timers(self):
        """Advance reproduction cooldown.  Called once per step."""
        if self._repro_timer > 0:
            self._repro_timer -= 1

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def get_info(self) -> dict:
        return {
            "id"           : self.agent_id,
            "species"      : self.species_type.value,
            "position"     : (round(self.position.x, 1), round(self.position.y, 1)),
            "energy"       : round(self.energy, 1),
            "age"          : self.age,
            "fitness"      : round(self.brain.fitness, 1),
            "offspring"    : self.offspring_born,
            "food_consumed": round(self.food_consumed, 1),
            "prey_captures": self.prey_captures,
            "hidden_size"  : self.brain.hidden_size,
            "recurrent"    : self.brain.has_recurrent,
        }
