"""Environment interface shared by the synthetic world and the LLM preference tables.

An environment knows the hidden truth (students' utilities, click probabilities) and hands
the policy only features and clicks. Queries are exogenous objects: a direction `z` for
the synthetic world, a question id for a table world.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .world import SyntheticWorld, sigmoid, sample_sphere, unit


class Environment:
    K: int
    d: int
    S: float
    beta: float
    name: str = "env"

    def sample_students(self, rng: np.random.Generator, N: int) -> list[Any]:
        raise NotImplementedError

    def features(self, q: Any) -> np.ndarray:
        raise NotImplementedError

    def utilities(self, student: Any, q: Any) -> np.ndarray:
        raise NotImplementedError

    def click_prob(self, student: Any, q: Any, i: int, j: int) -> float:
        raise NotImplementedError

    def rating(self, student: Any, q: Any, p: int, u: float) -> float:
        """Numerical-feedback reference mode: noisy utility of one shown persona on [0,1].
        `u` is a uniform in [0,1) used to derive the noise (common random numbers)."""
        raise NotImplementedError

    def student_type(self, student: Any) -> int:
        return -1

    def query_stream(self, kind: str, rng: np.random.Generator, student: Any, b: int, **kw) -> "QueryStream":
        raise NotImplementedError


class QueryStream:
    """Produces the exogenous query at round t. May look at the policy (adaptive adversary)
    and at the hidden student (it is the environment's adversary, not the learner's)."""

    def next(self, t: int, policy) -> Any:
        raise NotImplementedError


# --------------------------------------------------------------------------- synthetic

class _PredrawnStream(QueryStream):
    def __init__(self, Z: np.ndarray):
        self.Z = Z

    def next(self, t, policy):
        return self.Z[t]


class _AdversaryStream(QueryStream):
    """Adaptive adversary: from C fresh candidates per round, ask the question on which the
    policy's currently predicted-best persona loses the most utility (for this student)."""

    def __init__(self, env: "SyntheticEnv", student, Zc: np.ndarray):
        self.env, self.w, self.Zc = env, student, Zc  # Zc (b, C, d)

    def next(self, t, policy):
        cands = self.Zc[t]
        best_loss, best_z = -1.0, cands[0]
        for z in cands:
            Phi = self.env.features(z)
            V = self.env.utilities(self.w, z)
            p_hat = policy.predict_best(Phi)
            loss = V.max() - V[p_hat]
            if loss > best_loss:
                best_loss, best_z = loss, z
        return best_z


@dataclass
class SyntheticEnv(Environment):
    world: SyntheticWorld
    misspecified_frac: float = 0.0     # fraction of students drawn outside the population (D6)
    rating_noise: float = 0.1
    link: str = "logit"                # choice model: "logit" (Bradley-Terry), "probit" or "lapse"
    lapse: float = 0.15                # probability of a uniformly random choice under link="lapse"

    def __post_init__(self):
        self.K, self.d, self.S, self.beta = self.world.K, self.world.d, self.world.S, self.world.beta
        self.name = "synthetic"
        if self.link not in ("logit", "probit", "lapse"):
            raise ValueError(self.link)

    def stated_favourite(self, student) -> int:
        """Onboarding reference: the persona this student would name as their favourite, taken as
        the most frequent true favourite over a fixed pool of questions (hidden information; used
        only by the explicit-elicitation baseline)."""
        Z = sample_sphere(np.random.default_rng(12345), self.d, 64)
        favs = [int(np.argmax(self.utilities(student, z))) for z in Z]
        return int(np.bincount(favs, minlength=self.K).argmax())

    def sample_students(self, rng, N):
        W, types = self.world.population.sample(rng, N)
        students = []
        for u in range(N):
            if self.misspecified_frac > 0 and rng.random() < self.misspecified_frac:
                students.append((self.world.population.sample_misspecified(rng), -2))
            else:
                students.append((W[u], int(types[u])))
        return students

    def features(self, q):
        return self.world.phi(q)

    def utilities(self, student, q):
        w, _ = student
        return self.world.utilities(w, self.world.phi(q))

    def click_prob(self, student, q, i, j):
        V = self.utilities(student, q)
        z = self.beta * (V[i] - V[j])
        if self.link == "probit":
            # probit link with the slope matched to the logistic at the origin (Phi(z/1.702) ~ sigma(z))
            from scipy.special import ndtr
            return float(ndtr(z / 1.702))
        if self.link == "lapse":
            return float((1.0 - self.lapse) * sigmoid(z) + 0.5 * self.lapse)
        return float(sigmoid(z))

    def rating(self, student, q, p, u):
        V = self.utilities(student, q)
        U = (V[p] + self.S) / (2 * self.S)
        from scipy.special import ndtri
        noise = self.rating_noise * ndtri(min(max(u, 1e-9), 1 - 1e-9))
        return float(np.clip(U + noise, 0.0, 1.0))

    def student_type(self, student):
        return student[1]

    def query_stream(self, kind, rng, student, b, *, rho: float = 0.9, pool_size: int = 3,
                     n_cand: int = 16, pool: Optional[np.ndarray] = None):
        d = self.d
        if kind == "iid":
            return _PredrawnStream(sample_sphere(rng, d, b))
        if kind == "chain":
            Z = np.zeros((b, d))
            Z[0] = sample_sphere(rng, d)
            for t in range(1, b):
                Z[t] = unit(rho * Z[t - 1] + np.sqrt(1 - rho**2) * sample_sphere(rng, d))
            return _PredrawnStream(Z)
        if kind == "pool":
            if pool is None:
                pool = sample_sphere(rng, d, pool_size)
            start = rng.integers(0, len(pool))
            return _PredrawnStream(pool[(start + np.arange(b)) % len(pool)])
        if kind == "adversary":
            return _AdversaryStream(self, student, sample_sphere(rng, d, b * n_cand).reshape(b, n_cand, d))
        raise ValueError(kind)


# --------------------------------------------------------------------------- table (LLM)

@dataclass
class TableEnv(Environment):
    """Environment backed by precomputed tables (LLM students).

    Phi:    (Q, K, d) features of the generated answers
    P:      (Sn, Q, K, K) click probability that profile s prefers persona i over j on question q
    U:      (Sn, Q, K) evaluation utilities on [0,1] (fitted from held-out LLM judgments)
    chains: optional list of question-id sequences (follow-up chains)
    """

    Phi: np.ndarray
    P: np.ndarray
    U: np.ndarray
    profile_weights: Optional[np.ndarray] = None
    chains: Optional[list[list[int]]] = None
    S: float = 1.0
    beta: float = 1.0
    rating_noise: float = 0.1

    def __post_init__(self):
        self.Q, self.K, self.d = self.Phi.shape
        self.Sn = self.P.shape[0]
        self.name = "table"
        if self.profile_weights is None:
            self.profile_weights = np.full(self.Sn, 1.0 / self.Sn)

    def sample_students(self, rng, N):
        return [int(s) for s in rng.choice(self.Sn, size=N, p=self.profile_weights)]

    def features(self, q):
        return self.Phi[int(q)]

    def utilities(self, student, q):
        return self.U[student, int(q)]

    def click_prob(self, student, q, i, j):
        return float(self.P[student, int(q), i, j])

    def rating(self, student, q, p, u):
        from scipy.special import ndtri
        noise = self.rating_noise * ndtri(min(max(u, 1e-9), 1 - 1e-9))
        return float(np.clip(self.U[student, int(q), p] + noise, 0.0, 1.0))

    def student_type(self, student):
        return int(student)

    def stated_favourite(self, student) -> int:
        """Onboarding reference: the persona the profile would name, taken as the most frequent
        per-question Borda winner of its choice table (hidden information; used only by the
        explicit-elicitation baseline)."""
        fav = self.P[student].sum(axis=2).argmax(axis=1)          # (Q,)
        return int(np.bincount(fav, minlength=self.K).argmax())

    def query_stream(self, kind, rng, student, b, *, pool_size: int = 3, n_cand: int = 8, pool=None):
        Q = self.Q
        if kind == "iid":
            return _PredrawnStream(rng.integers(0, Q, size=b))
        if kind == "chain":
            if not self.chains:
                return _PredrawnStream(rng.integers(0, Q, size=b))
            seq = []
            while len(seq) < b:
                seq.extend(self.chains[rng.integers(0, len(self.chains))])
            return _PredrawnStream(np.array(seq[:b]))
        if kind == "pool":
            if pool is None:
                pool = rng.choice(Q, size=min(pool_size, Q), replace=False)
            start = rng.integers(0, len(pool))
            return _PredrawnStream(pool[(start + np.arange(b)) % len(pool)])
        if kind == "adversary":
            cands = rng.integers(0, Q, size=(b, n_cand))
            env = self

            class _Adv(QueryStream):
                def next(self, t, policy):
                    best, bq = -1.0, cands[t][0]
                    for q in cands[t]:
                        V = env.utilities(student, q)
                        p_hat = policy.predict_best(env.features(q))
                        loss = V.max() - V[p_hat]
                        if loss > best:
                            best, bq = loss, q
                    return bq

            return _Adv()
        raise ValueError(kind)
