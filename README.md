# EA-NN — Predator / Prey Neural Evolution Simulation

A predator-prey ecosystem where every agent is its own evolving neural network.
Herbivores learn to forage and evade; carnivores learn to hunt. Natural selection
drives the evolution of both populations across generations — no hand-coded
behaviour, only selection pressure.

---

## Quick Start

```bash
git clone <repository-url>
cd EA-NN
pip install -r requirements.txt

# Browser visualisation (recommended)
python main.py --web
# Open: http://localhost:5000

# Headless generational mode (20 generations × 750 steps each)
python main.py --generations 20

# Continuous mode (no GA resets, just ongoing evolution)
python main.py --steps 2000
```

---

## How It Works

### Agent loop

Each agent owns one neural network brain. Every simulation step:

1. **Sense** — 24 sensor values are assembled from the environment
2. **Decide** — the network runs a forward pass producing 4 outputs
3. **Act** — the agent moves, eats or hunts if in range, and optionally reproduces

Agents accumulate fitness by eating, surviving, and reproducing. At the end of
each generation the genetic algorithm selects the fittest brains, breeds
offspring via crossover and mutation, and resets the population.

### Neural network

| Property | Value |
|---|---|
| Architecture | Input → Hidden (tanh) → Output (sigmoid) |
| Input neurons | 24 |
| Hidden neurons | 12 – 32 (evolvable per-agent) |
| Output neurons | 4 |
| Recurrent connections | 70% of agents (short-term memory via `h_state`) |

Hidden layer size is itself an evolvable trait — structural mutations can grow or
shrink it while preserving the overlapping weights.

### Sensor layout (24 inputs)

| Index | Sensor |
|---|---|
| [0] | Energy level (0 – 1) |
| [1] | Age (0 – 1, normalised to 1 500 steps) |
| [2] | Reproduction readiness (0 / 1) |
| [3 – 8] | 3 × nearest food source: distance, angle |
| [9 – 14] | 3 × nearest threat / prey: distance, angle |
| [15] | X boundary proximity (0 = wall, 1 = centre) |
| [16] | Y boundary proximity |
| [17] | Time since last food — hunger signal (0 – 1) |
| [18 – 23] | 3 × nearest same-species companion: distance, angle |

Inputs [18 – 23] give each agent spatial awareness of its own species.
For carnivores this is the prerequisite for pack-hunting strategies; for
herbivores it enables herding. Neither behaviour is hard-coded — both must
emerge through selection pressure over many generations.

### Fitness signals

| Event | Fitness added |
|---|---|
| Each step alive | +0.05 (rewards longevity / successful evasion) |
| Energy gained from food or kill | +energy × 0.5 |
| Offspring born | +30.0 |

### Genetic algorithm

At the end of each generation:

- Top **20%** of each species carried over as elites (weights copied, state reset)
- Remaining slots filled by **tournament selection** (k = 4) + **crossover** (75%) or asexual copy
- All offspring receive **weight mutation** (~15% of weights perturbed)
- A species that goes fully extinct is **re-seeded** from random weights

### Species parameters

| Parameter | Herbivore | Carnivore |
|---|---|---|
| Starting energy | 100 | 120 |
| Max energy | 150 | 200 |
| Idle energy decay | 0.30 / step | 0.35 / step |
| Starvation kicks in after | — | 100 steps without food |
| Starvation extra decay | — | +0.65 / step (total 1.0 / step) |
| Speed | 2.0 units / step | 2.6 units / step |
| Vision range | 20 units | 25 units |
| Interaction range | 2.5 (eat food) | 5.0 (hunt prey) |
| Reproduction energy threshold | 80 | 120 |
| Reproduction cost | 35 | 45 |
| Reproduction cooldown | 120 steps | 160 steps |

---

## CLI reference

```bash
python main.py                                  # continuous, 500 steps
python main.py --steps 2000                     # custom step count
python main.py --generations 20                 # GA generational mode
python main.py --generations 50 --steps-per-gen 750
python main.py --web                            # browser visualisation on :5000
python main.py --web --port 8080                # custom port
```

---

## Web interface

Start with `python main.py --web` then open `http://localhost:5000`.

**Canvas** — live simulation. Green circles = herbivores, red triangles = carnivores,
amber squares = food. Click any agent to open the inspector panel.

**Agent inspector** — real-time readout for the selected agent:
- Energy bar and biological stats
- All 24 sensor input activations
- Recurrent hidden state values
- 4 output values with labels
- Live neural network weight diagram (connections coloured by sign and magnitude)

**Population chart** — rolling 120-step history of herbivore / carnivore counts.

**Speed slider** — 6 discrete presets:

| Position | Rate | Use |
|---|---|---|
| 1 | 1 / s | Watch individual decisions |
| 2 | 5 / s | |
| 3 | 10 / s | Default |
| 4 | 20 / s | |
| 5 | 50 / s | |
| 6 | MAX | As fast as the CPU allows |

---

## Project structure

```
EA-NN/
├── src/
│   ├── core/
│   │   └── ecosystem.py          # Environment, Food, Position, SpeciesType
│   ├── neural/
│   │   ├── neural_network.py     # NeuralNetwork, NetworkConfig
│   │   └── neural_agents.py      # NeuralAgent, SensorSystem
│   ├── evolution/
│   │   └── genetic_evolution.py  # GeneticAlgorithm, GAConfig
│   └── visualization/
│       └── web_server.py         # Flask + SocketIO server, HTML/CSS/JS
└── main.py                       # Entry point — continuous, generational, web modes
```

---

## Requirements

```
numpy>=1.21.0
flask>=2.0.0
flask-socketio>=5.0.0
```

---

## A note on emergent behaviour

Emergent strategies (pack hunting, prey fleeing, herding) require sustained
selection pressure over many generations — typically 50 – 100+ before patterns
stabilise in the gene pool. The sensor design makes them representable:

- **Pack hunting** — carnivores sense the positions of up to 3 companions ([18–23])
- **Prey tracking** — 70% of agents have recurrent connections; the hidden state
  persists across steps, allowing an agent to "remember" a target
- **Fleeing** — herbivores sense up to 3 carnivores ([9–14]) and receive a direct
  fitness reward (+0.05) for every step they remain alive

Run `python main.py --generations 100` to give evolution enough runway.
