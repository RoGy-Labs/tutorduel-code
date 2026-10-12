"""Baselines that do not use a linear logistic model."""
from __future__ import annotations

from typing import Optional

import numpy as np

from persona_bench.protocol import Policy


class RandomPairs(Policy):
    name = "random"

    def select(self, Phi):
        i, j = self.rng.choice(self.K, size=2, replace=False)
        return int(i), int(j)


class MaxDiameter(Policy):
    """Two most separated feature vectors."""
    name = "max_diameter"

    def select(self, Phi):
        D = np.linalg.norm(Phi[:, None, :] - Phi[None, :, :], axis=2)
        iu = np.triu_indices(self.K, 1)
        k = np.argmax(D[iu])
        return int(iu[0][k]), int(iu[1][k])


class CountHeuristic(Policy):
    name = "counts"

    def reset(self, rng, context):
        super().reset(rng, context)
        self.pop = np.zeros(self.K)

    def start_student(self):
        self.wins = np.zeros(self.K)
        self.shown = np.zeros(self.K, dtype=bool)
        self.t = 0
        self._last = None

    def _pop_order(self):
        return np.argsort(-(self.pop + 1e-6 * self.rng.random(self.K)))

    def select(self, Phi):
        order = self._pop_order()
        if self.t == 0:
            if self.pop.sum() < 2:
                i, j = self.rng.choice(self.K, size=2, replace=False)
            else:
                i, j = order[0], order[1]
        else:
            i = int(np.argmax(self.wins + 1e-6 * self.rng.random(self.K)))
            cands = [int(p) for p in order if p != i and not self.shown[p]]
            if not cands:
                cands = [int(p) for p in order if p != i]
            j = cands[0]
        i, j = int(i), int(j)
        self.shown[i] = self.shown[j] = True
        self._last = (i, j)
        return i, j

    def update(self, outcome):
        i, j = self._last
        self.wins[i if outcome == 1 else j] += 1
        self.t += 1

    def end_student(self):
        if self.t > 0:
            self.pop[int(np.argmax(self.wins + 1e-6 * self.rng.random(self.K)))] += 1

    def predict_best(self, Phi):
        return int(np.argmax(self.wins)) if self.t > 0 else int(np.argmax(self.pop))


class OnboardingElicitation(Policy):
    name = "onboarding"

    def reset(self, rng, context):
        super().reset(rng, context)
        self.stated: Optional[int] = None

    def set_stated(self, p: int):
        self.stated = int(p)

    def start_student(self):
        self.wins = np.zeros(self.K)
        self._last = None

    def select(self, Phi):
        i = self.stated if self.stated is not None else int(self.rng.integers(self.K))
        others = [p for p in range(self.K) if p != i]
        if self.wins[others].max() <= 0:
            j = int(self.rng.choice(others))
        else:
            w = self.wins.copy()
            w[i] = -np.inf
            j = int(np.argmax(w))
        self._last = (i, j)
        return i, j

    def update(self, outcome):
        i, j = self._last
        self.wins[i if outcome == 1 else j] += 1

    def predict_best(self, Phi):
        return self.stated if self.stated is not None else 0
