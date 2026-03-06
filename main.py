#!/usr/bin/env python3
"""
EA-NN  —  Predator / Prey Neural Evolution Simulation
======================================================

Each agent (herbivore or carnivore) is its own neural network.
Natural selection via predation and starvation drives the weights
toward effective foraging / hunting behaviour over generations.

Usage
-----
    python main.py                      # headless simulation, 500 steps
    python main.py --steps 2000         # custom step count
    python main.py --generations 20     # run 20 discrete GA generations
    python main.py --web                # start browser visualisation
    python main.py --web --port 5001    # custom port

Architecture
------------
  - Environment   (src/core/ecosystem.py)       — world, food, step loop
  - NeuralNetwork (src/neural/neural_network.py) — feedforward + optional recurrent brain
  - NeuralAgent   (src/neural/neural_agents.py)  — sense → decide → act
  - GeneticAlgorithm (src/evolution/genetic_evolution.py) — inter-generation selection
"""

import argparse
import os
import random
import sys

import numpy as np

# ── project root on the path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.ecosystem       import Environment, SpeciesType, Position
from src.neural.neural_agents import NeuralAgent
from src.neural.neural_network import NetworkConfig
from src.evolution.genetic_evolution import GeneticAlgorithm, GAConfig


# ======================================================================= #
# Population seeding                                                       #
# ======================================================================= #

def seed_population(env: Environment, herbivores: int = 20, carnivores: int = 8):
    """Place the initial generation of agents at random positions."""
    cfg = NetworkConfig()
    agents = []

    for _ in range(herbivores):
        pos = Position(
            random.uniform(10.0, env.width  - 10.0),
            random.uniform(10.0, env.height - 10.0),
        )
        agents.append(NeuralAgent(SpeciesType.HERBIVORE, pos, env.next_agent_id(), config=cfg))

    for _ in range(carnivores):
        pos = Position(
            random.uniform(10.0, env.width  - 10.0),
            random.uniform(10.0, env.height - 10.0),
        )
        agents.append(NeuralAgent(SpeciesType.CARNIVORE, pos, env.next_agent_id(), config=cfg))

    env.add_agents(agents)


# ======================================================================= #
# Simulation modes                                                         #
# ======================================================================= #

def run_continuous(steps: int, print_every: int = 100):
    """
    Run the simulation continuously for *steps* steps.
    Agents reproduce in-simulation; there is no explicit generation reset.
    """
    print("=== EA-NN  |  Continuous mode ===")
    env = Environment(width=100, height=100)
    seed_population(env)

    _print_header()

    for step in range(1, steps + 1):
        env.step()

        if step % print_every == 0:
            _print_stats(step, env)

        if not env.agents:
            print(f"[step {step}] Total extinction — ending early.")
            break

    print("\n=== Final state ===")
    _print_stats(steps, env)


def run_generational(total_generations: int, steps_per_gen: int = 750, print_every: int = 150):
    """
    Run discrete generations: simulate for *steps_per_gen* steps, then
    apply the genetic algorithm, then repeat.
    """
    print(f"=== EA-NN  |  Generational mode  ({total_generations} gen x {steps_per_gen} steps) ===")

    env   = Environment(width=100, height=100)
    ga    = GeneticAlgorithm(GAConfig())

    seed_population(env)
    _print_header()

    history = []   # list of per-gen stat dicts for trend tracking

    for gen in range(1, total_generations + 1):
        print(f"\n-- Generation {gen} --")

        for step in range(1, steps_per_gen + 1):
            env.step()

            if step % print_every == 0:
                _print_stats((gen - 1) * steps_per_gen + step, env)

            if not env.agents:
                print(f"[gen {gen}, step {step}] Total extinction.")
                break

        # Collect rich learning stats before evolving
        stats = _collect_gen_stats(env.agents, gen)
        history.append(stats)
        _print_gen_summary(stats, history)

        # Evolve
        if env.agents:
            env.agents = ga.evolve(env.agents, env)
        else:
            print("  -> Seeding from scratch (total extinction).")
            seed_population(env)

    print("\n=== Simulation complete ===")
    _print_learning_summary(history)


def run_web(steps_per_gen: int = 750, port: int = 5000):
    """Start the browser-based visualisation."""
    try:
        from src.visualization.web_server import EcosystemWebServer
    except ImportError:
        print("Web server not available. Run without --web.")
        return

    env = Environment(width=100, height=100)
    ga  = GeneticAlgorithm(GAConfig())
    seed_population(env)

    class _Wrapper:
        """Thin adapter that the web server expects."""
        def __init__(self, environment, genetic_algorithm):
            self.env        = environment
            self.ga         = genetic_algorithm
            self.step_count = 0
            self._gen_step  = 0
            self._steps_per_gen = steps_per_gen

        def step(self):
            self.env.step()
            self.step_count += 1
            self._gen_step  += 1
            if self._gen_step >= self._steps_per_gen and self.env.agents:
                self.env.agents = self.ga.evolve(self.env.agents, self.env)
                self._gen_step  = 0

        def update(self):
            self.step()

        def get_stats(self):
            stats = self.env.get_stats()
            stats["generation"] = self.ga.generation
            return _jsonify(stats)

        def get_agent_data(self):
            herbs = {"x": [], "y": [], "fitness": [], "ids": []}
            carns = {"x": [], "y": [], "fitness": [], "ids": []}
            food  = {"x": [], "y": []}

            for a in self.env.agents:
                bucket = herbs if a.species_type == SpeciesType.HERBIVORE else carns
                bucket["x"].append(float(a.position.x))
                bucket["y"].append(float(a.position.y))
                bucket["fitness"].append(float(a.brain.fitness))
                bucket["ids"].append(a.agent_id)

            for f in self.env.food_sources:
                if f.is_available:
                    food["x"].append(float(f.position.x))
                    food["y"].append(float(f.position.y))

            return {"herbivores": herbs, "carnivores": carns, "food": food}

    wrapper = _Wrapper(env, ga)
    server  = EcosystemWebServer(wrapper)

    print(f"Open your browser at  http://localhost:{port}")
    print("Press Ctrl-C to stop.")
    server.start_server()


# ======================================================================= #
# Utilities                                                                #
# ======================================================================= #

def _collect_gen_stats(agents, gen_num: int) -> dict:
    """Collect rich behavioural + fitness stats from agents at end of a generation."""
    herbs = [a for a in agents if a.species_type == SpeciesType.HERBIVORE]
    carns = [a for a in agents if a.species_type == SpeciesType.CARNIVORE]

    def _stats(group):
        if not group:
            return None
        fitnesses = [a.brain.fitness for a in group]
        return {
            "count"       : len(group),
            "avg_fitness" : float(np.mean(fitnesses)),
            "max_fitness" : float(np.max(fitnesses)),
            "avg_food"    : float(np.mean([a.food_consumed  for a in group])),
            "avg_offspring": float(np.mean([a.offspring_born for a in group])),
            "avg_prey"    : float(np.mean([a.prey_captures  for a in group])),
            "avg_energy"  : float(np.mean([a.energy         for a in group])),
            "avg_hidden"  : float(np.mean([a.brain.hidden_size for a in group])),
        }

    return {"gen": gen_num, "herbivores": _stats(herbs), "carnivores": _stats(carns)}


def _trend(current: float, previous: float) -> str:
    """Return a one-char trend indicator comparing current to previous value."""
    if previous == 0:
        return "~"
    delta = (current - previous) / abs(previous)
    if delta >  0.05: return "^"
    if delta < -0.05: return "v"
    return "~"


def _print_gen_summary(stats: dict, history: list):
    """Print a readable end-of-generation learning summary with trend arrows."""
    prev = history[-2] if len(history) >= 2 else None

    def _fmt(sp_key):
        s = stats[sp_key]
        if s is None:
            return "  EXTINCT"
        p = prev[sp_key] if prev else None
        tf = _trend(s["avg_fitness"],  p["avg_fitness"])  if p else "~"
        te = _trend(s["avg_energy"],   p["avg_energy"])   if p else "~"
        to = _trend(s["avg_offspring"],p["avg_offspring"]) if p else "~"
        tp = _trend(s["avg_prey"],     p["avg_prey"])      if p else "~"

        base = (f"  n={s['count']:>2}  "
                f"fit avg={s['avg_fitness']:>7.1f}{tf}  max={s['max_fitness']:>7.1f}  "
                f"food={s['avg_food']:>5.1f}  "
                f"energy={s['avg_energy']:>5.1f}{te}  "
                f"offspring={s['avg_offspring']:>4.1f}{to}  "
                f"hidden={s['avg_hidden']:>4.1f}")
        if sp_key == "carnivores":
            base += f"  prey={s['avg_prey']:>4.1f}{tp}"
        return base

    print(f"  [H] {_fmt('herbivores')}")
    print(f"  [C] {_fmt('carnivores')}")


def _print_learning_summary(history: list):
    """Print a compact generation-over-generation fitness table at the end."""
    if not history:
        return
    print("\n=== Learning Progress Summary ===")
    print(f"{'Gen':>4}  {'H-pop':>5}  {'H-fit-avg':>9}  {'H-fit-max':>9}  "
          f"{'C-pop':>5}  {'C-fit-avg':>9}  {'C-fit-max':>9}  {'C-prey-avg':>10}")
    print("-" * 75)
    for s in history:
        h = s["herbivores"]
        c = s["carnivores"]
        hpop = h["count"]       if h else 0
        hfa  = h["avg_fitness"] if h else 0.0
        hfm  = h["max_fitness"] if h else 0.0
        cpop = c["count"]       if c else 0
        cfa  = c["avg_fitness"] if c else 0.0
        cfm  = c["max_fitness"] if c else 0.0
        cpr  = c["avg_prey"]    if c else 0.0
        print(f"{s['gen']:>4}  {hpop:>5}  {hfa:>9.1f}  {hfm:>9.1f}  "
              f"{cpop:>5}  {cfa:>9.1f}  {cfm:>9.1f}  {cpr:>10.2f}")


def _print_header():
    print(f"{'Step':>6}  {'Herbs':>5}  {'Carns':>5}  {'Food':>5}  "
          f"{'AvgHE':>7}  {'AvgCE':>7}")
    print("-" * 50)


def _print_stats(step: int, env: Environment):
    s = env.get_stats()
    print(
        f"{step:>6}  {s['herbivores']:>5}  {s['carnivores']:>5}  "
        f"{s['available_food']:>5}  "
        f"{s['avg_herbivore_energy']:>7.1f}  {s['avg_carnivore_energy']:>7.1f}"
    )


def _jsonify(obj):
    """Recursively convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ======================================================================= #
# Entry point                                                              #
# ======================================================================= #

def main():
    parser = argparse.ArgumentParser(description="EA-NN predator/prey simulation")
    parser.add_argument("--steps",       type=int,  default=500,  help="Steps for continuous mode")
    parser.add_argument("--generations", type=int,  default=0,    help="Number of GA generations (0 = continuous)")
    parser.add_argument("--steps-per-gen", type=int, default=750, help="Simulation steps per generation")
    parser.add_argument("--web",         action="store_true",     help="Start browser visualisation")
    parser.add_argument("--port",        type=int,  default=5000, help="Web server port")
    args = parser.parse_args()

    if args.web:
        run_web(steps_per_gen=args.steps_per_gen, port=args.port)
    elif args.generations > 0:
        run_generational(args.generations, steps_per_gen=args.steps_per_gen)
    else:
        run_continuous(args.steps)


if __name__ == "__main__":
    main()
