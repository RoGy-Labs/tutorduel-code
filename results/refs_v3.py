"""DIAGNOSTIC reference policies (hidden information; NOT admissible): `refx:typelib:<rule>`.

Same input as `ref:typelib` (per-type mean click and utility tables of the design roster) but with pair rules that value
information about the type while paying strong regret. They bound what an identification-aware rule could reach if the
population were known: rule = greedy (= ref:typelib), kg (one-step lookahead, greedy continuation), rollout (Monte Carlo
one-step lookahead with a greedy rollout that keeps learning), ids (information-directed sampling on the type), lts and
lucb (latent-bandit Thompson sampling and UCB over the types, Hong et al. 2020; c = log-likelihood-ratio margin of lucb)."""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from persona_bench.protocol import Policy  # noqa: E402
from persona_bench.registry import register  # noqa: E402


def _H(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class TypeLibX(Policy):
    population = "reference"

    def __init__(self, name, rule="greedy", n_roll=64, c=3.0):
        self.name, self.rule, self.n_roll, self.c = name, rule, n_roll, c

    def reset(self, rng, context):
        super().reset(rng, context)
        lib = context["typelib"]
        self.P, self.U, self.w0 = np.clip(lib["P"], 1e-6, 1 - 1e-6), lib["U"], lib["weights"]
        self.b = context["b"]
        K = self.K
        self.pairs = np.array(list(combinations(range(K), 2)))                       # (15, 2)
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        self.R = 2 * self.U.max(1, keepdims=True) - self.U[:, i] - self.U[:, j]      # (M, 15) strong regret per type
        self.Pp = self.P[:, i, j]                                                     # (M, 15) P(first chosen)
        self.logPp1, self.logPp0 = np.log(self.Pp), np.log(1 - self.Pp)

    def start_student(self):
        self.logw = np.log(self.w0); self.t = 0

    def _w(self):
        w = np.exp(self.logw - self.logw.max()); return w / w.sum()

    def _greedy_pair(self, w):
        return int(np.argmin(w @ self.R))

    def select(self, Phi):
        w = self._w(); rem = self.b - self.t - 1
        rule = self.rule
        if rule == "greedy" or rem <= 0:
            p = self._greedy_pair(w)
        elif rule == "kg":
            imm = w @ self.R                                                          # (15,)
            pbar = w @ self.Pp
            w1 = w[:, None] * self.Pp / pbar[None]; w0 = w[:, None] * (1 - self.Pp) / (1 - pbar)[None]   # (M,15)
            v1 = (w1.T @ self.R).min(1); v0 = (w0.T @ self.R).min(1)                  # (15,) future greedy regret per round
            p = int(np.argmin(imm + rem * (pbar * v1 + (1 - pbar) * v0)))
        elif rule == "lts":
            # latent-bandit Thompson sampling over the types: greedy pair of one type drawn from the posterior
            k = int(self.rng.choice(len(w), p=w))
            p = int(np.argmin(self.R[k] + 1e-9 * self.rng.random(len(self.pairs))))
        elif rule == "lucb":
            # latent-bandit UCB: lowest regret under the most optimistic plausible type (posterior within exp(-c) of the top)
            lw = np.log(np.maximum(w, 1e-300))
            p = int(np.argmin(self.R[lw >= lw.max() - self.c].min(axis=0) + 1e-9 * self.rng.random(len(self.pairs))))
        elif rule == "ids":
            imm = w @ self.R
            pbar = w @ self.Pp
            info = _H(pbar) - w @ _H(self.Pp)
            p = int(np.argmin(imm ** 2 / np.maximum(info, 1e-9)))
        elif rule == "rollout":
            p = self._rollout(w, rem)
        else:
            raise ValueError(rule)
        self._last = (int(self.pairs[p, 0]), int(self.pairs[p, 1]), p)
        return self._last[0], self._last[1]

    def _rollout(self, w, rem):
        S, M, NP = self.n_roll, len(w), len(self.pairs)
        rng = self.rng
        k = rng.choice(M, size=S, p=w)                                               # sampled true types (shared across candidates)
        u = rng.random(size=(rem + 1, S))
        cost = np.zeros(NP)
        logw = np.broadcast_to(self.logw, (NP, S, M)).copy()                         # belief per (candidate, sample)
        # first step: candidate pair p
        y = (u[0][None, :] < self.Pp[k].T)                                           # (NP, S): Pp[k] is (S, NP)
        logw += np.where(y[..., None], self.logPp1.T[:, None, :], self.logPp0.T[:, None, :])
        cost += self.R[k].T.mean(1)                                                  # (NP,)
        for r in range(rem):
            wr = np.exp(logw - logw.max(-1, keepdims=True)); wr /= wr.sum(-1, keepdims=True)
            g = np.argmin(wr @ self.R, axis=-1)                                      # (NP, S) greedy pair
            cost += self.R[k[None, :], g].mean(1)
            y = u[r + 1][None, :] < self.Pp[k[None, :], g]
            logw += np.where(y[..., None], self.logPp1[:, g].transpose(1, 2, 0), self.logPp0[:, g].transpose(1, 2, 0))
        return int(np.argmin(cost))

    def update(self, outcome):
        i, j, p = self._last
        self.logw = self.logw + (self.logPp1[:, p] if outcome else self.logPp0[:, p])
        self.t += 1

    def diagnostics(self):
        return {"k_hat": int(np.argmax(self.logw))}


def _make(spec):
    parts = spec.split(":")
    kw = {}
    for p in parts[3:]:
        if p.startswith("n"): kw["n_roll"] = int(p[1:])
        elif p.startswith("c"): kw["c"] = float(p[1:])
    return TypeLibX(name=spec, rule=parts[2] if len(parts) > 2 else "greedy", **kw)


register("refx", _make)
