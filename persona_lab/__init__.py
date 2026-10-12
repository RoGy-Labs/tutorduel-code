"""persona_lab: shared inference code and comparison methods (CODuel itself is in persona_v3).

Spec strings: 'lab:<family>[:opt...]'. Import this package before `run_sequence`.
  lab:pop                        pooled baseline: one ridge logistic model on every completed session, two highest scores
  lab:metats / hierts / adats    Meta-TS, HierTS, AdaTS adapted to pairwise choices (metats.py)
hb.py, kde.py and infer.py hold the Laplace / mixture code that CODuel's population learner (persona_v3/mix.py) builds on.
"""
from __future__ import annotations

from persona_bench.registry import register

from .hb import HBMix
from .kde import KDEPrior
from .metats import PriorTS, make_prior_ts
from .pop import PooledAug, make_pop

FAMILIES = {"pop": make_pop, "metats": make_prior_ts, "hierts": make_prior_ts, "adats": make_prior_ts}


def make_lab(spec: str):
    parts = spec.split(":")
    fam = parts[1] if len(parts) > 1 else ""
    if fam not in FAMILIES:
        raise ValueError(f"unknown lab family {fam!r}; known {sorted(FAMILIES)}")
    return FAMILIES[fam](spec)


register("lab", make_lab)

__all__ = ["HBMix", "KDEPrior", "PooledAug", "PriorTS", "make_lab"]
