"""Population-level pooled rules (reference members of the portfolio, no within-session update):
one ridge-logistic vector on all completed sessions over augmented features [psi, aug * onehot(persona)],
pair = two highest predicted utilities. Optionally a Thompson-sampled vector for exploration."""
from __future__ import annotations

import numpy as np

from persona_bench.protocol import Policy

from .hb import parse_opts
from .infer import laplace_fit


class PooledAug(Policy):
    DEFAULTS = dict(aug=1.0, eps=0.0, lam_scale=1.0, ts=False, refit_every=1, explore=0)

    def __init__(self, name="lab:pop", **opts):
        self.name = name
        self.opt = dict(self.DEFAULTS); self.opt.update(opts)

    def reset(self, rng, context):
        super().reset(rng, context)
        self.S, self.beta = context["S"], context["beta"]
        self.D = self.d + (self.K if self.opt["aug"] > 0 else 0)
        self.lam = self.opt["lam_scale"] * self.d / max(self.S ** 2, 1e-9)
        self.w_pool = np.zeros(self.D); self.H = self.lam * np.eye(self.D)
        self.log_X = []; self.log_o = []
        self._sx = []; self._so = []; self._last = None; self.n = 0
        self.G = self.lam * np.eye(self.D); self.Ginv = np.linalg.inv(self.G); self.t_global = 0

    def _aug(self, Phi):
        a = self.opt["aug"]
        if a <= 0:
            return Phi
        return np.concatenate([Phi, a * np.eye(self.K)], axis=1) / np.sqrt(1.0 + a * a)

    def start_student(self):
        self._sx, self._so = [], []
        self.w_use = self.w_pool
        if self.opt["ts"] and self.n > 0:
            L = np.linalg.cholesky(np.linalg.inv(self.H) + 1e-9 * np.eye(self.D))
            self.w_use = self.w_pool + L @ self.rng.normal(size=self.D)

    def select(self, Phi):
        Phi = self._aug(Phi)
        V = Phi @ self.w_use
        order = np.argsort(-V)
        i, j = int(order[0]), int(order[1])
        if self.opt["explore"] > 0 and self.t_global < self.opt["explore"]:
            iu = np.triu_indices(self.K, 1)
            D = self.beta * (Phi[iu[0]] - Phi[iu[1]])
            cov = np.einsum("pd,de,pe->p", D, self.Ginv, D)
            k = int(np.argmax(cov)); i, j = int(iu[0][k]), int(iu[1][k])
        self._last = (i, j, Phi)
        return i, j

    def update(self, outcome):
        i, j, Phi = self._last
        x = self.beta * (Phi[i] - Phi[j])
        self._sx.append(x); self._so.append(float(outcome))
        self.t_global += 1
        if self.opt["explore"] > 0:
            self.G += 0.25 * np.outer(x, x); self.Ginv = np.linalg.inv(self.G)

    def end_student(self):
        if not self._sx:
            return
        self.log_X.append(np.array(self._sx)); self.log_o.append(np.array(self._so))
        self.n += 1
        if self.n % self.opt["refit_every"] == 0:
            X = np.concatenate(self.log_X)[None]; o = np.concatenate(self.log_o)[None]
            w, H, _, _ = laplace_fit(X, o, np.zeros((1, self.D)), (self.lam * np.eye(self.D))[None], w0=self.w_pool[None], eps=self.opt["eps"])
            self.w_pool, self.H = w[0], H[0]

    def predict_best(self, Phi):
        return int(np.argmax(self._aug(Phi) @ self.w_pool))


def make_pop(spec: str):
    parts = spec.split(":")
    opts = parse_opts(parts[2:], PooledAug.DEFAULTS)
    return PooledAug(name=spec, **opts)
