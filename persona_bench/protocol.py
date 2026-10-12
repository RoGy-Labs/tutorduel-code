"""Policy contract for the persona-dueling benchmark.

A third-party algorithm implements `Policy` and is scored by `run_sequence`.
The policy never receives true utilities, types, or click probabilities.
"""
from __future__ import annotations

import numpy as np


class Policy:
    """Sequential-student pair selector.

    Lifecycle per run:
        reset(rng, context)   once, before student 1
        start_student()       a student arrives
        select(Phi) -> (i, j) two distinct persona indices
        update(o)             click (1 = first index won)
        end_student()         the student leaves

    `context` contains K, d, S, beta, b, N, and optionally oracle (reference policies only).
    """

    name: str = "policy"

    def reset(self, rng: np.random.Generator, context: dict) -> None:
        self.rng = rng
        self.ctx = context
        self.K, self.d = context["K"], context["d"]

    def start_student(self) -> None:
        pass

    def select(self, Phi: np.ndarray) -> tuple[int, int]:
        raise NotImplementedError

    def update(self, outcome: int) -> None:
        pass

    def end_student(self) -> None:
        pass

    def predict_best(self, Phi: np.ndarray) -> int:
        return 0

    def select_single(self, Phi: np.ndarray) -> int:
        return int(self.rng.integers(0, self.K))

    def update_rating(self, y: float) -> None:
        pass

    def diagnostics(self) -> dict:
        return {}
