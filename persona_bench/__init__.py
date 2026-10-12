"""Persona-dueling benchmark: environments, scoring, Policy API, baselines, leaderboard.

A third-party algorithm implements `Policy` and is run with `run_sequence`.
Import `persona_bench.baselines` (done here) to register shipped comparators.
"""
from __future__ import annotations

from . import baselines  # noqa: F401  registers none/oracle/random/colstim/coldb/pooled
from .env import Environment, SyntheticEnv, TableEnv
from .leaderboard import build_leaderboard, write_leaderboard
from .protocol import Policy
from .registry import make_policy, register
from .dataset import load_shared, load_types, shared_env, types_env, types_scale
from .replay import (
    ArrivalEnv,
    calibrate_S,
    filter_arrivals,
    no_transfer_oracle,
    relabel_personas,
    subset_arrivals,
    type_oracle,
)
from .report import excess_curves, i_conv, result_row, summary_bundle
from .runner import RunResult, run_sequence
from .world import make_world

__all__ = [
    "Policy",
    "make_policy",
    "register",
    "run_sequence",
    "RunResult",
    "Environment",
    "SyntheticEnv",
    "TableEnv",
    "ArrivalEnv",
    "make_world",
    "calibrate_S",
    "filter_arrivals",
    "no_transfer_oracle",
    "relabel_personas",
    "subset_arrivals",
    "type_oracle",
    "excess_curves",
    "load_types",
    "load_shared",
    "types_env",
    "shared_env",
    "types_scale",
    "i_conv",
    "result_row",
    "summary_bundle",
    "build_leaderboard",
    "write_leaderboard",
]
