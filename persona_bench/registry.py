"""Register and build policies by spec string (e.g. 'none:emax').

The benchmark registers baselines. Algorithm packages call `register` on import.
`persona_bench` never imports those packages.
"""
from __future__ import annotations

from typing import Callable

from .protocol import Policy

_PREFIX: dict[str, Callable[[str], Policy]] = {}


def register(prefix: str, factory: Callable[[str], Policy]) -> None:
    _PREFIX[prefix] = factory


def make_policy(spec: str | Policy) -> Policy:
    if isinstance(spec, Policy):
        return spec
    head = spec.split(":")[0]
    if head not in _PREFIX:
        known = ", ".join(sorted(_PREFIX))
        raise ValueError(
            f"unknown policy {spec!r} (head {head!r}). Registered: {known or '(none)'}. "
            "Import your algorithm package before make_policy."
        )
    return _PREFIX[head](spec)
