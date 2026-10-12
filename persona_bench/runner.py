"""Sequential-student runner with common random numbers.

For a given (env, seed): the N students, their query streams, and the click uniforms
are drawn once and shared by every policy. Differences between policies are the pairs
they show, not luck.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .env import Environment
from .protocol import Policy
from .registry import make_policy


@dataclass
class RunResult:
    policy: str
    regret: np.ndarray          # (N, b) strong regret 2U*-U1-U2, native utility scale
    weak_regret: np.ndarray     # (N, b) U* - max(U1, U2), descriptive
    floor: np.ndarray           # (N, b) U* - U^(2), best-pair floor g_t
    regretful: np.ndarray       # (N, b) 1 if the best persona was not shown
    pairs: np.ndarray           # (N, b, 2)
    outcomes: np.ndarray        # (N, b)
    types: np.ndarray           # (N,) student type (-1 if none, -2 misspecified)
    to01: float                 # multiply regret by this to get the [0,1] utility scale
    seconds: float
    diagnostics: list[dict] = field(default_factory=list)
    queries: Optional[list] = None
    avg_regret: Optional[np.ndarray] = None   # regret of the clicked answer

    @property
    def total(self) -> float:
        return float(self.regret.sum())

    @property
    def per_student(self) -> np.ndarray:
        return self.regret.sum(axis=1)

    @property
    def avg_total(self) -> float:
        return float(self.avg_regret.sum()) if self.avg_regret is not None else float("nan")

    @property
    def G_T(self) -> float:
        return float(self.floor.sum())


def _hash_name(name: str) -> int:
    import zlib
    return int(zlib.crc32(name.encode()) % (1 << 31))


def _second_max(V: np.ndarray) -> float:
    if V.size < 2:
        return float(V.max())
    return float(np.partition(V, -2)[-2])


def run_sequence(env: Environment, policies: list[str | Policy], N: int, b: int, seed: int,
                 query_kind: str = "iid", feedback: str = "ab", query_kw: Optional[dict] = None,
                 oracle: Optional[dict] = None, keep_queries: bool = False,
                 extra_context: Optional[dict] = None) -> dict[str, RunResult]:
    """Run every policy on the same sequence of N students with b rounds each."""
    query_kw = dict(query_kw or {})
    ss = np.random.SeedSequence([seed, 1])
    s_students, s_queries, s_clicks, s_pol = ss.spawn(4)
    rng_students = np.random.default_rng(s_students)
    students = env.sample_students(rng_students, N)
    uniforms = np.random.default_rng(s_clicks).random(size=(N, b))
    to01 = 1.0 / (2 * env.S) if env.name == "synthetic" else 1.0
    shared_pool = None
    if query_kind == "pool":
        rng_pool = np.random.default_rng(s_queries.spawn(1)[0])
        if env.name == "synthetic":
            from .world import sample_sphere
            shared_pool = sample_sphere(rng_pool, env.d, query_kw.get("pool_size", 3))
        else:
            shared_pool = rng_pool.choice(env.Q, size=min(query_kw.get("pool_size", 3), env.Q), replace=False)
        query_kw["pool"] = shared_pool

    results = {}
    for pspec in policies:
        pol = make_policy(pspec) if not isinstance(pspec, Policy) else pspec
        t0 = time.time()
        ctx = {"K": env.K, "d": env.d, "S": env.S, "beta": env.beta, "b": b, "N": N,
               "feedback": feedback, "rating_noise": getattr(env, "rating_noise", 0.1)}
        if oracle is not None and getattr(pol, "population", None) == "oracle":
            ctx["oracle"] = oracle
        if extra_context:
            ctx.update(extra_context)
        rng_pol = np.random.default_rng([int(s_pol.generate_state(1)[0]), _hash_name(pol.name)])
        pol.reset(rng_pol, ctx)
        rng_q = np.random.default_rng(s_queries)
        regret = np.zeros((N, b))
        weak = np.zeros((N, b))
        floor = np.zeros((N, b))
        avg_regret = np.zeros((N, b))
        regretful = np.zeros((N, b), dtype=np.int8)
        pairs = np.zeros((N, b, 2), dtype=np.int16)
        outcomes = np.zeros((N, b), dtype=np.int8)
        types = np.array([env.student_type(s) for s in students])
        diags = []
        queries = [] if keep_queries else None
        for u, student in enumerate(students):
            stream = env.query_stream(query_kind, rng_q, student, b, **query_kw)
            pol.start_student()
            if hasattr(pol, "set_stated") and hasattr(env, "stated_favourite"):
                pol.set_stated(env.stated_favourite(student))
            for t in range(b):
                q = stream.next(t, pol)
                Phi = env.features(q)
                V = env.utilities(student, q)
                vmax = float(V.max())
                g_t = vmax - _second_max(V)
                if feedback == "ab":
                    i, j = pol.select(Phi)
                    assert i != j and 0 <= i < env.K and 0 <= j < env.K, (i, j)
                    P = env.click_prob(student, q, i, j)
                    o = int(uniforms[u, t] < P)
                    pol.update(o)
                    regret[u, t] = 2 * vmax - V[i] - V[j]
                    weak[u, t] = vmax - max(V[i], V[j])
                    floor[u, t] = g_t
                    avg_regret[u, t] = vmax - (V[i] if o == 1 else V[j])
                    regretful[u, t] = int(not (np.isclose(V[i], vmax) or np.isclose(V[j], vmax)))
                    pairs[u, t] = (i, j)
                    outcomes[u, t] = o
                else:
                    p = pol.select_single(Phi)
                    y = env.rating(student, q, p, uniforms[u, t])
                    pol.update_rating(y)
                    regret[u, t] = vmax - V[p]
                    weak[u, t] = regret[u, t]
                    floor[u, t] = 0.0
                    avg_regret[u, t] = regret[u, t]
                    regretful[u, t] = int(not np.isclose(V[p], vmax))
                    pairs[u, t] = (p, -1)
                if keep_queries:
                    queries.append(q)
            pol.end_student()
            diags.append(pol.diagnostics())
        results[pol.name] = RunResult(
            pol.name, regret, weak, floor, regretful, pairs, outcomes, types, to01,
            time.time() - t0, diags, queries, avg_regret,
        )
    return results
