"""RUCB, Relative Upper Confidence Bound (Zoghi, Whiteson, Munos and de Rijke, ICML 2014), pooled across users.

A population-level, NON-contextual dueling bandit over the K personas: one wins matrix W (K x K) for the whole run,
shared by every user and question, ignoring the persona-question features. Algorithm 1 of the paper:
  * U_ij = W_ij / (W_ij + W_ji) + sqrt(alpha ln t / (W_ij + W_ji)), with x/0 := 1 and U_ii = 1/2;
  * candidates C = {c : U_cj >= 1/2 for all j}; champion c drawn from C (uniformly from [K] when C is empty; when C has
    several arms, the hypothesised best arm B, set whenever C is a singleton, is drawn with probability 1/2 and the
    others share the rest);
  * challenger d = argmax_{j != c} U_jc (ties at random); the pair (c, d) is displayed and W updated with the click.
alpha > 1/2 is the exploration constant of the paper (0.51 in its experiments). Spec "rucb[:aX]".
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from persona_bench.protocol import Policy


class RUCB(Policy):
    name = "rucb"

    def __init__(self, alpha: float = 0.51, name: Optional[str] = None):
        self.alpha = alpha
        self.name = name or "rucb"

    def reset(self, rng, context):
        super().reset(rng, context)
        self.W = np.zeros((self.K, self.K))
        self.t = 0
        self.B: Optional[int] = None
        self._last = None

    def select(self, Phi):
        K = self.K
        self.t += 1
        N = self.W + self.W.T
        with np.errstate(divide="ignore", invalid="ignore"):
            U = np.where(N > 0, self.W / np.maximum(N, 1e-12) + np.sqrt(self.alpha * np.log(max(self.t, 2)) / np.maximum(N, 1e-12)), 1.0)
        np.fill_diagonal(U, 0.5)
        C = np.nonzero((U >= 0.5).all(axis=1))[0]
        if len(C) == 0:
            c = int(self.rng.integers(K))
        elif len(C) == 1:
            c = int(C[0]); self.B = c
        else:
            p = np.ones(len(C))
            if self.B is not None and self.B in C:
                p = np.where(C == self.B, 0.5, 0.5 / (len(C) - 1))
            c = int(self.rng.choice(C, p=p / p.sum()))
        score = U[:, c] + 1e-9 * self.rng.random(K)
        score[c] = -np.inf
        d = int(np.argmax(score))
        self._last = (c, d)
        return c, d

    def update(self, outcome):
        c, d = self._last
        if outcome:
            self.W[c, d] += 1
        else:
            self.W[d, c] += 1

    def predict_best(self, Phi):
        N = self.W + self.W.T
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(N > 0, self.W / np.maximum(N, 1e-12), 0.5)
        return int(np.argmax(P.sum(axis=1)))

    def diagnostics(self):
        return {"B": -1 if self.B is None else int(self.B)}


def make_rucb(spec: str) -> Policy:
    parts = spec.split(":")
    alpha = 0.51
    for p in parts[1:]:
        if p.startswith("a"):
            alpha = float(p[1:])
        else:
            raise ValueError(spec)
    return RUCB(alpha=alpha, name=spec)
