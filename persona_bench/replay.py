"""LLM replay worlds: private-pack env, S calibration, arrival filters, oracles."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Optional, Sequence

import numpy as np

from .env import Environment, TableEnv, _PredrawnStream


@dataclass
class ArrivalEnv(Environment):
    """Option-2 replay: each arrival has a private stream of questions.

    Phi:  (Q, K, d) features of every private question, globally indexed.
    P:    (N, b, K, K) click probabilities on that arrival's b questions.
    U:    (N, b, K) utilities on [0, 1].
    qids: (N, b) global question ids for each arrival's stream.
    """

    Phi: np.ndarray
    P: np.ndarray
    U: np.ndarray
    qids: np.ndarray
    S: float = 1.0
    beta: float = 1.0
    rating_noise: float = 0.1
    q_to_ut: dict = field(default_factory=dict, repr=False)
    tau: float = 1.0                       # click sharpness: click logits are multiplied by tau (P itself stays raw)
    click: str = "raw"                     # "raw": clicks follow P; "cbt": clicks follow the Bradley-Terry table Pcbt
    Pcbt: Optional[np.ndarray] = None      # (N, b, K, K) sigma(v_i - v_j), consistent with the utilities
    profiles: Optional[list] = None        # (N,) profile id of each arrival (hidden; for reports and references only)
    arrival_ids: Optional[np.ndarray] = None   # (N,) index of each arrival in the unfiltered roster

    def __post_init__(self):
        self.Q, self.K, self.d = self.Phi.shape
        self.N, self.b_priv, K2, K3 = self.P.shape
        assert K2 == K3 == self.K
        assert self.U.shape == (self.N, self.b_priv, self.K)
        assert self.qids.shape == (self.N, self.b_priv)
        self.Sn = self.N
        self.name = "table"
        if self.click not in ("raw", "cbt"):
            raise ValueError(self.click)
        if self.click == "cbt" and self.Pcbt is None:
            raise ValueError("click='cbt' needs the Pcbt table")
        from .utility import sharpen
        self._Pclick = sharpen(self.P if self.click == "raw" else self.Pcbt, float(self.tau))
        if self.arrival_ids is None:
            self.arrival_ids = np.arange(self.N)
        self.q_to_ut = {}
        for u in range(self.N):
            for t in range(self.b_priv):
                q = int(self.qids[u, t])
                self.q_to_ut[q] = (u, t)

    def _slot(self, student, q):
        u, t = self.q_to_ut[int(q)]
        if u != int(student):
            raise KeyError(f"question {q} does not belong to arrival {student}")
        return u, t

    def sample_students(self, rng, N):
        if N > self.N:
            raise ValueError(f"asked for {N} arrivals, roster has {self.N}")
        return [int(i) for i in rng.permutation(self.N)[:N]]

    def features(self, q):
        return self.Phi[int(q)]

    def utilities(self, student, q):
        u, t = self._slot(student, q)
        return self.U[u, t]

    def click_prob(self, student, q, i, j):
        u, t = self._slot(student, q)
        return float(self._Pclick[u, t, i, j])

    def rating(self, student, q, p, u):
        from scipy.special import ndtri
        uu, t = self._slot(student, q)
        noise = self.rating_noise * ndtri(min(max(u, 1e-9), 1 - 1e-9))
        return float(np.clip(self.U[uu, t, p] + noise, 0.0, 1.0))

    def student_type(self, student):
        return int(student)

    def type_ids(self) -> np.ndarray:
        """(N,) integer type of each arrival (index into the sorted profile names); -1 if profiles are unknown."""
        if self.profiles is None:
            return np.full(self.N, -1)
        names = sorted(set(self.profiles))
        return np.array([names.index(p) for p in self.profiles])

    def stated_favourite(self, student) -> int:
        fav = self.P[int(student)].sum(axis=2).argmax(axis=1)
        return int(np.bincount(fav, minlength=self.K).argmax())

    def query_stream(self, kind, rng, student, b, **kw):
        seq = np.asarray(self.qids[int(student), np.arange(b) % self.b_priv], dtype=int)   # b > b_priv repeats the pack
        return _PredrawnStream(seq)


def filter_arrivals(
    P: np.ndarray,
    U: np.ndarray,
    qids: np.ndarray,
    profiles: Sequence[str],
    keep: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep arrivals whose profile id is in `keep`. Returns P, U, qids, mask."""
    keep_set = set(keep)
    mask = np.array([p in keep_set for p in profiles], dtype=bool)
    if not mask.any():
        raise ValueError(f"no arrivals match {sorted(keep_set)}")
    return P[mask], U[mask], qids[mask], mask


def linear_fits(P: np.ndarray, features_fn, K: int, d: int) -> np.ndarray:
    """Per-arrival least-squares w on logit P. features_fn(s, t) -> (K, d)."""
    Sn = P.shape[0]
    b = P.shape[1]
    atoms = np.zeros((Sn, d))
    for s in range(Sn):
        X, y = [], []
        for t in range(b):
            Phi = features_fn(s, t)
            for i, j in combinations(range(K), 2):
                X.append(Phi[i] - Phi[j])
                y.append(P[s, t, i, j])
        p = np.clip(np.asarray(y), 1e-3, 1 - 1e-3)
        z = np.log(p / (1 - p))
        w, *_ = np.linalg.lstsq(np.asarray(X), z, rcond=None)
        atoms[s] = w
    return atoms


def linear_oracle_from_atoms(atoms: np.ndarray) -> dict:
    n = len(atoms)
    return {"kind": "types", "atoms": atoms, "weights": np.full(n, 1.0 / n)}


def calibrate_S(atoms: np.ndarray) -> float:
    """Median Euclidean norm of per-student linear fits."""
    return float(np.median(np.linalg.norm(np.asarray(atoms), axis=1)))


def no_transfer_oracle(d: int, S: float) -> dict:
    """Isotropic prior matching the no-transfer scale."""
    return {"kind": "gaussian", "mu": np.zeros(d), "Sigma": (S ** 2 / d) * np.eye(d)}


def arrival_oracle(env: ArrivalEnv) -> tuple[dict, np.ndarray]:
    """Linear-fit types oracle on an ArrivalEnv, plus the atom matrix."""
    def feat(s, t):
        return env.features(int(env.qids[s, t]))
    atoms = linear_fits(env.P, feat, env.K, env.d)
    return linear_oracle_from_atoms(atoms), atoms


def table_oracle(env: TableEnv) -> tuple[dict, np.ndarray]:
    Sn, Q, K, d = env.P.shape[0], env.Q, env.K, env.d
    atoms = np.zeros((Sn, d))
    for s in range(Sn):
        X, y = [], []
        for q in range(Q):
            Phi = env.Phi[q]
            for i, j in combinations(range(K), 2):
                X.append(Phi[i] - Phi[j])
                y.append(env.P[s, q, i, j])
        p = np.clip(np.asarray(y), 1e-3, 1 - 1e-3)
        z = np.log(p / (1 - p))
        w, *_ = np.linalg.lstsq(np.asarray(X), z, rcond=None)
        atoms[s] = w
    return linear_oracle_from_atoms(atoms), atoms


def relabel_personas(env: ArrivalEnv, perm) -> ArrivalEnv:
    """Same world with persona indices permuted (new persona k is old persona perm[k]). Nothing a policy may use
    changes except the labels, so an admissible policy must be indifferent in distribution."""
    perm = np.asarray(perm)
    pc = None if env.Pcbt is None else env.Pcbt[:, :, perm][:, :, :, perm]
    return ArrivalEnv(Phi=env.Phi[:, perm], P=env.P[:, :, perm][:, :, :, perm], U=env.U[:, :, perm], qids=env.qids, S=env.S, beta=env.beta,
                      rating_noise=env.rating_noise, tau=env.tau, click=env.click, Pcbt=pc, profiles=env.profiles, arrival_ids=env.arrival_ids)


def subset_arrivals(env: ArrivalEnv, idx) -> ArrivalEnv:
    idx = np.asarray(idx)
    return ArrivalEnv(Phi=env.Phi, P=env.P[idx], U=env.U[idx], qids=env.qids[idx], S=env.S, beta=env.beta, rating_noise=env.rating_noise,
                      tau=env.tau, click=env.click, Pcbt=None if env.Pcbt is None else env.Pcbt[idx],
                      profiles=None if env.profiles is None else [env.profiles[i] for i in idx], arrival_ids=env.arrival_ids[idx])


def type_oracle(fit_env: ArrivalEnv, aug: float = 1.0, lam: float = 1.0) -> dict:
    """Type-level reference prior: one atom per profile type, fitted by ridge logistic regression of the click table
    on [style features, aug * persona identity]. `fit_env` must be the design (dev) roster, never the roster being
    evaluated. Returns {"kind": "types", "atoms" (M, d + K), "weights", "names", "aug"}."""
    from scipy.special import expit
    assert fit_env.profiles is not None
    K, d = fit_env.K, fit_env.d
    names = sorted(set(fit_env.profiles))
    atoms = []
    for nm in names:
        X, y = [], []
        for u in [i for i, p in enumerate(fit_env.profiles) if p == nm]:
            for t in range(fit_env.b_priv):
                ph = np.concatenate([fit_env.Phi[int(fit_env.qids[u, t])], aug * np.eye(K)], axis=1) / np.sqrt(1 + aug * aug)
                for i, j in combinations(range(K), 2):
                    X.append(ph[i] - ph[j]); y.append(fit_env.P[u, t, i, j])
        X, y = np.asarray(X), np.asarray(y)
        w = np.zeros(d + K)
        for _ in range(50):
            sg = expit(X @ w)
            w -= np.linalg.solve((X * (sg * (1 - sg))[:, None]).T @ X + lam * np.eye(d + K), X.T @ (sg - y) + lam * w)
        atoms.append(w)
    counts = np.array([fit_env.profiles.count(nm) for nm in names], dtype=float)
    return {"kind": "types", "atoms": np.asarray(atoms), "weights": counts / counts.sum(), "names": names, "aug": aug}
