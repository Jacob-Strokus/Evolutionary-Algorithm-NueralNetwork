"""
Neural network for agent decision-making.

Architecture: input → hidden (tanh) → output (sigmoid)
Optional recurrent self-connections on the hidden layer give agents short-term memory.
Hidden layer size is an evolvable trait — crossover handles mismatched sizes cleanly.
"""

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NetworkConfig:
    input_size            : int   = 24    # see SensorSystem for layout
    output_size           : int   = 4     # move_x, move_y, speed, reproduce
    min_hidden            : int   = 12   # larger floor for more complex policies
    max_hidden            : int   = 32   # larger ceiling for emergent behaviors
    mutation_rate         : float = 0.15  # fraction of weights mutated per call
    mutation_strength     : float = 0.20  # std-dev of weight perturbation
    structure_mutation_rate: float = 0.03  # chance to change hidden-layer size per call
    recurrent_prob        : float = 0.70  # most agents get memory (was 0.40)


class NeuralNetwork:
    """
    A feedforward neural network with optional recurrent hidden-state connections.

    All new networks are created with randomly-sized hidden layers (within the
    config bounds) and a random decision about recurrence.  Structural mutations
    can change the hidden-layer size mid-evolution while preserving learned weights.
    """

    def __init__(
        self,
        config       : NetworkConfig,
        hidden_size  : Optional[int]  = None,
        has_recurrent: Optional[bool] = None,
    ):
        self.config        = config
        self.hidden_size   = hidden_size   if hidden_size   is not None else random.randint(config.min_hidden, config.max_hidden)
        self.has_recurrent = has_recurrent if has_recurrent is not None else (random.random() < config.recurrent_prob)

        # Performance bookkeeping
        self.fitness   = 0.0
        self.decisions = 0

        self._init_weights()

    # ------------------------------------------------------------------ #
    # Weight initialisation                                                #
    # ------------------------------------------------------------------ #

    def _init_weights(self):
        cfg = self.config
        h   = self.hidden_size

        self.W1 = np.random.randn(cfg.input_size, h) * 0.4
        self.b1 = np.zeros(h)
        self.W2 = np.random.randn(h, cfg.output_size) * 0.4
        self.b2 = np.zeros(cfg.output_size)

        self.Wr: Optional[np.ndarray] = (
            np.random.randn(h, h) * 0.2 if self.has_recurrent else None
        )
        self.h_state = np.zeros(h)  # recurrent hidden state

    # ------------------------------------------------------------------ #
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run one forward pass.  Input is automatically padded / truncated."""
        x = np.asarray(x, dtype=float)
        n = self.config.input_size
        if x.shape[0] < n:
            x = np.pad(x, (0, n - x.shape[0]))
        elif x.shape[0] > n:
            x = x[:n]

        pre = x @ self.W1 + self.b1
        if self.has_recurrent and self.Wr is not None:
            pre = pre + self.h_state @ self.Wr

        self.h_state = np.tanh(np.clip(pre, -10.0, 10.0))

        out = self.h_state @ self.W2 + self.b2
        self.decisions += 1
        return 1.0 / (1.0 + np.exp(-np.clip(out, -10.0, 10.0)))  # sigmoid

    def reset_state(self):
        """Clear the recurrent hidden state (call between independent episodes)."""
        self.h_state = np.zeros(self.hidden_size)

    # ------------------------------------------------------------------ #
    # Evolution operators                                                  #
    # ------------------------------------------------------------------ #

    def mutate(self):
        """Perturb weights in-place and occasionally change network structure."""
        rate     = self.config.mutation_rate
        strength = self.config.mutation_strength

        for w in (self.W1, self.b1, self.W2, self.b2):
            mask = np.random.random(w.shape) < rate
            w   += mask * np.random.randn(*w.shape) * strength

        if self.Wr is not None:
            mask     = np.random.random(self.Wr.shape) < rate
            self.Wr += mask * np.random.randn(*self.Wr.shape) * (strength * 0.5)

        if random.random() < self.config.structure_mutation_rate:
            self._mutate_structure()

    def _mutate_structure(self):
        """Randomly resize the hidden layer, preserving as many weights as possible."""
        new_h = random.randint(self.config.min_hidden, self.config.max_hidden)
        if new_h == self.hidden_size:
            return

        old_h = self.hidden_size
        copy  = min(old_h, new_h)

        new_W1 = np.random.randn(self.config.input_size, new_h) * 0.2
        new_W1[:, :copy] = self.W1[:, :copy]

        new_b1 = np.zeros(new_h)
        new_b1[:copy] = self.b1[:copy]

        new_W2 = np.random.randn(new_h, self.config.output_size) * 0.2
        new_W2[:copy, :] = self.W2[:copy, :]

        if self.Wr is not None:
            new_Wr = np.random.randn(new_h, new_h) * 0.1
            new_Wr[:copy, :copy] = self.Wr[:copy, :copy]
            self.Wr = new_Wr

        self.W1          = new_W1
        self.b1          = new_b1
        self.W2          = new_W2
        self.hidden_size = new_h
        self.h_state     = np.zeros(new_h)

        # Occasionally toggle recurrence when structure changes
        if random.random() < 0.15:
            self.has_recurrent = not self.has_recurrent
            if self.has_recurrent:
                self.Wr = np.random.randn(new_h, new_h) * 0.2
            else:
                self.Wr = None

    def crossover(self, other: "NeuralNetwork") -> "NeuralNetwork":
        """
        Produce an offspring by blending weight matrices from two parents.

        When the hidden layers differ in size, only the overlapping slice is
        mixed; the remaining neurons are initialised randomly, giving the
        offspring its own unique genes in that region.
        """
        # Inherit network structure from one parent at random
        if random.random() < 0.5:
            h, rec = self.hidden_size, self.has_recurrent
        else:
            h, rec = other.hidden_size, other.has_recurrent

        child = NeuralNetwork(self.config, hidden_size=h, has_recurrent=rec)

        min_h = min(self.hidden_size, other.hidden_size, h)

        # --- W1 (input → hidden) ---
        for i in range(self.config.input_size):
            for j in range(min_h):
                child.W1[i, j] = self.W1[i, j] if random.random() < 0.5 else other.W1[i, j]

        # --- b1 ---
        for i in range(min_h):
            child.b1[i] = self.b1[i] if random.random() < 0.5 else other.b1[i]

        # --- W2 (hidden → output) ---
        for i in range(min_h):
            for j in range(self.config.output_size):
                child.W2[i, j] = self.W2[i, j] if random.random() < 0.5 else other.W2[i, j]

        # --- b2 ---
        for i in range(self.config.output_size):
            child.b2[i] = self.b2[i] if random.random() < 0.5 else other.b2[i]

        # --- Wr (recurrent) ---
        if child.Wr is not None and self.Wr is not None and other.Wr is not None:
            for i in range(min_h):
                for j in range(min_h):
                    child.Wr[i, j] = self.Wr[i, j] if random.random() < 0.5 else other.Wr[i, j]

        return child

    def copy(self) -> "NeuralNetwork":
        """Return an exact deep copy of this network (fitness is not copied)."""
        clone              = NeuralNetwork(self.config, self.hidden_size, self.has_recurrent)
        clone.W1           = self.W1.copy()
        clone.b1           = self.b1.copy()
        clone.W2           = self.W2.copy()
        clone.b2           = self.b2.copy()
        clone.Wr           = self.Wr.copy() if self.Wr is not None else None
        clone.h_state      = self.h_state.copy()
        return clone
