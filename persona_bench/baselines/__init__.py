"""Shipped baselines. Import this package to register them with persona_bench.make_policy."""
from __future__ import annotations

from persona_bench.baselines.coldb import ColdbPolicy, make_coldb
from persona_bench.baselines.laplace_policy import ColstimDueling, LaplaceDueling, PooledW
from persona_bench.baselines.rucb import RUCB, make_rucb
from persona_bench.baselines.simple import CountHeuristic, MaxDiameter, OnboardingElicitation, RandomPairs
from persona_bench.protocol import Policy
from persona_bench.registry import register


def _parse_kw(parts: list[str]) -> dict:
    kw = {}
    for extra in parts:
        if extra == "noadapt":
            kw["adapt"] = False
        elif extra == "rt":
            kw["random_ties"] = True
        elif extra.startswith("fut"):
            kw["future"] = float(extra[3:])
        elif extra.startswith("c"):
            kw["ucb_c"] = float(extra[1:])
        else:
            raise ValueError(f"unknown baseline option {extra}")
    return kw


def make_baseline(spec: str) -> Policy:
    if spec == "random":
        return RandomPairs()
    if spec == "max_diameter":
        return MaxDiameter()
    if spec == "counts":
        return CountHeuristic()
    if spec == "onboarding":
        return OnboardingElicitation()
    if spec == "pooled":
        return PooledW()
    if spec.startswith("colstim"):
        # 'colstim', 'colstim:c0.3', 'colstim:c0.3:l3:g1:t3' (c = exploration width, l = ridge multiplier,
        # g = scale of CoLSTIM's Gumbel perturbation of the leader (0 = none), t = trim threshold of the perturbation)
        c, lm, g, cth = 1.0, 1.0, 0.0, 3.0
        for part in spec.split(":")[1:]:
            if part.startswith("c"):
                c = float(part[1:])
            elif part.startswith("l"):
                lm = float(part[1:])
            elif part.startswith("g"):
                g = float(part[1:])
            elif part.startswith("t"):
                cth = float(part[1:])
            else:
                raise ValueError(spec)
        return ColstimDueling(c=c, name=spec, lam_mult=lm, g=g, cth=cth)
    if spec.split(":")[0] in ("coldb", "coldbc", "coldbm", "coldbl"):
        return make_coldb(spec)
    if spec.split(":")[0] == "rucb":
        return make_rucb(spec)
    parts = spec.split(":")
    population, rule = parts[0], parts[1]
    if population not in ("none", "oracle"):
        raise ValueError(spec)
    kw = _parse_kw(parts[2:])
    return LaplaceDueling(population=population, rule=rule, name=spec, **kw)


for _prefix in ("random", "max_diameter", "counts", "onboarding", "pooled", "colstim",
                "coldb", "coldbc", "coldbm", "coldbl", "rucb", "none", "oracle"):
    register(_prefix, make_baseline)

__all__ = [
    "RandomPairs", "MaxDiameter", "CountHeuristic", "OnboardingElicitation",
    "PooledW", "ColstimDueling", "LaplaceDueling", "ColdbPolicy", "RUCB",
    "make_baseline",
]
