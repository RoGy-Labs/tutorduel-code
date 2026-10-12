"""persona_v3: the proposed method CODuel. Import before `run_sequence`.

Spec prefixes: 'v3:' = the population learner with its evidence gate (CODuel on the simulator, whose features contain no rater
score); 'qo:' = the same learner with the rater's score recalibrated from completed sessions (CODuel on the tutoring benchmark,
persona_v3/recal.py)."""
from __future__ import annotations

from persona_bench.registry import register

from .mix import MixV3
from .policy import GatedV3, make_v3
from .recal import GatedQO, make_qo

register("v3", make_v3)
register("qo", make_qo)

__all__ = ["MixV3", "GatedV3", "make_v3", "GatedQO", "make_qo"]
