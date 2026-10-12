"""Synthetic world: feature map phi(q, p) and user populations.

Everything is drawn from `numpy.random.Generator` streams spawned from a single
`SeedSequence`, so changing one knob does not silently change unrelated draws.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- helpers

def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


def unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalise along `axis`; zero vectors map to e_0."""
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    out = np.divide(x, n, out=np.zeros_like(x), where=n > 0)
    zero = np.squeeze(n <= 0, axis=axis)
    if np.any(zero):
        idx = np.nonzero(zero)
        e0 = np.zeros(x.shape[axis])
        e0[0] = 1.0
        out[idx] = e0
    return out


def sample_sphere(rng: np.random.Generator, d: int, size: Optional[int] = None) -> np.ndarray:
    if size is None:
        x = rng.normal(size=d)
        return x / np.linalg.norm(x)
    x = rng.normal(size=(size, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def haar_rotation(rng: np.random.Generator, d: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def project_ball(w: np.ndarray, S: float) -> np.ndarray:
    n = np.linalg.norm(w, axis=-1, keepdims=True)
    return np.where(n > S, S * w / np.maximum(n, 1e-12), w)


# --------------------------------------------------------------------------- populations

@dataclass
class Population:
    """User population. `kind` in {"types", "gaussian", "lowrank"}."""

    kind: str
    d: int
    S: float
    atoms: Optional[np.ndarray] = None       # (M, d) for types
    mu: Optional[np.ndarray] = None          # (d,) for gaussian / lowrank
    Sigma: Optional[np.ndarray] = None       # (d, d) for gaussian / lowrank
    subspace: Optional[np.ndarray] = None    # (d, r) for lowrank
    params: dict = field(default_factory=dict)

    @property
    def M(self) -> int:
        return 0 if self.atoms is None else self.atoms.shape[0]

    def sample(self, rng: np.random.Generator, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return (W (n,d), types (n,) with -1 when no type)."""
        if self.kind == "types":
            c = rng.integers(0, self.M, size=n)
            return self.atoms[c].copy(), c
        L = np.linalg.cholesky(self.Sigma + 1e-12 * np.eye(self.d))
        W = self.mu[None, :] + rng.normal(size=(n, self.d)) @ L.T
        return project_ball(W, self.S), -np.ones(n, dtype=int)

    def sample_misspecified(self, rng: np.random.Generator, eta: float = 0.5) -> np.ndarray:
        """A user the population model does not cover (D6): blend of two types, or
        a vector orthogonal to the cloud's centre."""
        if self.kind == "types":
            c, c2 = rng.choice(self.M, size=2, replace=False)
            w = (1 - eta) * self.atoms[c] + eta * self.atoms[c2]
            return self.S * w / np.linalg.norm(w)
        mu = self.mu
        w_typ = min(self.S, np.linalg.norm(mu)) if np.linalg.norm(mu) > 0 else self.S
        q, _ = np.linalg.qr(np.column_stack([mu / max(np.linalg.norm(mu), 1e-12), rng.normal(size=(self.d, self.d - 1))]))
        v = q[:, 1]
        return w_typ * v

    def true_prior(self) -> dict:
        """What an oracle policy is allowed to know."""
        if self.kind == "types":
            return {"kind": "types", "atoms": self.atoms.copy(), "weights": np.full(self.M, 1.0 / self.M)}
        return {"kind": "gaussian", "mu": self.mu.copy(), "Sigma": self.Sigma.copy()}


def make_type_atoms(rng: np.random.Generator, d: int, M: int, theta_deg: float, S: float) -> np.ndarray:
    """M atoms with equal pairwise angle theta (spec §4.1)."""
    th = np.deg2rad(theta_deg)
    Q = haar_rotation(rng, d)
    if d >= M + 1:
        u = Q[:, 0]
        V = Q[:, 1 : M + 1]
        alpha = np.arccos(np.sqrt(max(np.cos(th), 0.0)))
        W = np.cos(alpha) * u[None, :] + np.sin(alpha) * V.T
        return S * W
    if d == M:
        V = Q[:, :M].T
        u = unit(V.sum(axis=0))
        c = np.cos(th)
        rhs = c / (1 - c) if c < 1 else 1e6
        # gamma^2 + 2 gamma / sqrt(M) = rhs
        gamma = max(0.0, -1 / np.sqrt(M) + np.sqrt(1 / M + rhs)) if th < np.pi / 2 else 0.0
        W = unit(gamma * u[None, :] + V)
        return S * W
    raise ValueError("need d >= M")


def make_gaussian(rng: np.random.Generator, d: int, s: float, S: float, rank: Optional[int] = None,
                  noise: float = 0.05) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Gaussian cloud (spec §4.2). With `rank=r`, Sigma has r dominant directions (shared
    low-rank structure) plus `noise^2 * s^2` isotropic jitter, and mu lies in the subspace."""
    if rank is None:
        mu = 0.5 * S * sample_sphere(rng, d)
        A = rng.normal(size=(d, d))
        S0 = A @ A.T + 0.3 * np.eye(d)
        S0 /= np.trace(S0) / d
        return mu, s**2 * S0, None
    U = haar_rotation(rng, d)[:, :rank]
    A = rng.normal(size=(rank, rank))
    C = A @ A.T + 0.3 * np.eye(rank)
    C /= np.trace(C) / rank
    Sigma = s**2 * (U @ C @ U.T + noise**2 * np.eye(d))
    c = sample_sphere(rng, rank)
    mu = 0.5 * S * (U @ c)
    return mu, Sigma, U


# --------------------------------------------------------------------------- world

@dataclass
class SyntheticWorld:
    d: int
    K: int
    S: float
    beta: float
    lam: float
    anchors: np.ndarray        # (K, d)
    B: np.ndarray              # (K, d, d)
    population: Population
    diagnostics: dict = field(default_factory=dict)
    seed: int = 0

    def phi(self, z: np.ndarray) -> np.ndarray:
        """Feature matrix (K, d) for a query direction z (d,)."""
        raw = (1.0 - self.lam) * self.anchors + self.lam * np.einsum("kij,j->ki", self.B, z)
        return unit(raw, axis=1)

    def phi_batch(self, Z: np.ndarray) -> np.ndarray:
        """(n, d) -> (n, K, d)."""
        raw = (1.0 - self.lam) * self.anchors[None] + self.lam * np.einsum("kij,nj->nki", self.B, Z)
        return unit(raw, axis=2)

    def utilities(self, w: np.ndarray, Phi: np.ndarray) -> np.ndarray:
        return np.clip(Phi @ w, -self.S, self.S)


def _diagnostics(world: SyntheticWorld, rng: np.random.Generator, n_diag: int = 256) -> dict:
    """Instance diagnostics (spec §6): disagreement rates, click separation eps, mean gap."""
    Z = sample_sphere(rng, world.d, n_diag)
    Phi = world.phi_batch(Z)                                     # (n, K, d)
    pop = world.population
    out = {"lambda": world.lam, "population": pop.kind, "beta": world.beta}
    if pop.kind == "types":
        V = np.einsum("nkd,md->nmk", Phi, pop.atoms)             # (n, M, K)
        best = V.argmax(axis=2)                                  # (n, M)
        M = pop.M
        out["query_disagreement_rate"] = float(np.mean((best != best[:, :1]).any(axis=1)))
        pair_dis = []
        eps_pairs = []
        eps_pairs_min = []
        for c in range(M):
            for c2 in range(c + 1, M):
                dis = best[:, c] != best[:, c2]
                pair_dis.append(dis.mean())
                if dis.any():
                    idx = np.nonzero(dis)[0]
                    pc, pc2 = best[idx, c], best[idx, c2]
                    vc = V[idx, c, :]
                    vc2 = V[idx, c2, :]
                    e = np.abs(
                        sigmoid(world.beta * (vc[np.arange(len(idx)), pc] - vc[np.arange(len(idx)), pc2]))
                        - sigmoid(world.beta * (vc2[np.arange(len(idx)), pc] - vc2[np.arange(len(idx)), pc2]))
                    )
                    eps_pairs.append(e.mean())
                    eps_pairs_min.append(e.min())
                else:
                    eps_pairs.append(0.0)
                    eps_pairs_min.append(0.0)
        out["type_pair_disagreement_rate"] = float(np.mean(pair_dis))
        out["eps"] = float(np.min(eps_pairs))            # min over type pairs of the MEAN over discriminating queries
        out["eps_min"] = float(np.min(eps_pairs_min))    # min over type pairs AND queries (Assumption 2 as stated)
        # mean gap for a typical (type) user
        Vs = np.sort(V, axis=2)
        out["mean_gap"] = float(np.mean(Vs[:, :, -1] - Vs[:, :, -2]))
        out["M"] = M
    else:
        W, _ = pop.sample(rng, 64)
        V = np.einsum("nkd,md->nmk", Phi, W)
        Vs = np.sort(V, axis=2)
        out["mean_gap"] = float(np.mean(Vs[:, :, -1] - Vs[:, :, -2]))
        best = V.argmax(axis=2)                                  # (n, 64)
        out["query_disagreement_rate"] = float(np.mean((best != best[:, :1]).any(axis=1)))
        # fraction of user pairs disagreeing on the favourite, averaged over queries
        eq = best[:, :, None] == best[:, None, :]
        out["type_pair_disagreement_rate"] = float(1.0 - (eq.sum() - eq.shape[0] * eq.shape[1]) / (eq.shape[0] * eq.shape[1] * (eq.shape[1] - 1)))
        out["eps"] = float("nan")
        out["M"] = 0
    return out


def make_world(seed: int, *, d: int = 8, K: int = 6, S: float = 1.0, beta: float = 1.0, lam: float = 1.0,
               population: str = "types", M: int = 4, theta_deg: float = 60.0, s: float = 0.3,
               rank: Optional[int] = None, max_resample: int = 64) -> SyntheticWorld:
    """Generate a world; resample (new anchors / frame, same knobs) until the disagreement
    diagnostics land in [0.05, 1] (spec §6.1)."""
    ss = np.random.SeedSequence(seed)
    diag_ok = False
    for attempt in range(max_resample + 1):
        # each attempt gets fresh, isolated streams
        s_feat, s_pop, s_diag = [np.random.default_rng(x) for x in ss.spawn(3)]
        for _ in range(attempt):  # burn so that attempts differ deterministically
            s_feat.normal(); s_pop.normal()
        anchors = sample_sphere(s_feat, d, K)
        B = s_feat.normal(scale=1.0 / np.sqrt(d), size=(K, d, d))
        if population == "types":
            atoms = make_type_atoms(s_pop, d, M, theta_deg, S)
            pop = Population("types", d, S, atoms=atoms, params={"M": M, "theta_deg": theta_deg})
        elif population == "gaussian":
            mu, Sigma, _ = make_gaussian(s_pop, d, s, S)
            pop = Population("gaussian", d, S, mu=mu, Sigma=Sigma, params={"s": s})
        elif population == "lowrank":
            r = rank if rank is not None else 2
            mu, Sigma, U = make_gaussian(s_pop, d, s, S, rank=r)
            pop = Population("lowrank", d, S, mu=mu, Sigma=Sigma, subspace=U, params={"s": s, "rank": r})
        else:
            raise ValueError(population)
        world = SyntheticWorld(d, K, S, beta, lam, anchors, B, pop, seed=seed)
        diag = _diagnostics(world, s_diag)
        diag_ok = not (diag["query_disagreement_rate"] < 0.05 and diag["type_pair_disagreement_rate"] < 0.05)
        if diag_ok:
            break
    diag["diagnostics_ok"] = bool(diag_ok)
    diag["resamples"] = attempt
    world.diagnostics = diag
    return world
