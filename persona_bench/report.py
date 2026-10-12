"""Reporting helpers: checkpoints, i_conv, per-run rows, paired summaries."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .metrics import mean_ci, summarize_runs
from .runner import RunResult

CHECKPOINTS = (60, 100, 300)
I_CONV_WINDOW = 20


def _to01(r: RunResult, arr: np.ndarray) -> np.ndarray:
    return arr * r.to01


def checkpoint_totals(per_student01: np.ndarray, checkpoints: tuple[int, ...] = CHECKPOINTS) -> dict:
    out = {}
    N = len(per_student01)
    csum = np.cumsum(per_student01)
    for k in checkpoints:
        if N >= k:
            out[f"total01_at_{k}"] = float(csum[k - 1])
    return out


def i_conv(
    learner_per_student: np.ndarray,
    oracle_per_student: np.ndarray,
    window: int = I_CONV_WINDOW,
) -> Optional[int]:
    """First 1-based student index i at which the trailing-`window` mean of the
    learner is <= the oracle's trailing mean on the same window. None if never.
    """
    a = np.asarray(learner_per_student, dtype=float)
    b = np.asarray(oracle_per_student, dtype=float)
    N = len(a)
    if N < window or len(b) != N:
        return None
    for i in range(window, N + 1):
        sl = slice(i - window, i)
        if float(a[sl].mean()) <= float(b[sl].mean()):
            return i
    return None


def excess_curves(r: RunResult) -> dict:
    """Regret in excess of the best-pair floor (strong regret can never go below U* - U^(2) per round).
    by_round01[t]: mean over students of the regret in round t; terciles split the arrival order in three."""
    reg, flo = r.regret * r.to01, r.floor * r.to01
    N = reg.shape[0]
    cuts = [0, N // 3, (2 * N) // 3, N]
    return {
        "excess01": float(reg.sum() - flo.sum()),
        "by_round01": [float(x) for x in reg.mean(axis=0)],
        "floor_by_round01": [float(x) for x in flo.mean(axis=0)],
        "excess_by_round_tercile01": [[float(x) for x in (reg[a:b] - flo[a:b]).mean(axis=0)] if b > a else [] for a, b in zip(cuts[:-1], cuts[1:])],
        "excess_per_student01": [float(x) for x in (reg - flo).sum(axis=1)],
    }


def result_row(r: RunResult, extra: Optional[dict] = None, checkpoints: tuple[int, ...] = CHECKPOINTS) -> dict:
    ps = r.per_student * r.to01
    N = len(ps)
    n10 = max(1, N // 10)
    n15 = max(1, int(0.15 * N))
    row = {
        "policy": r.policy,
        "N": N,
        "total01": r.total * r.to01,
        "per_student01": float(ps.mean()),
        "first_decile01": float(ps[:n10].mean()),
        "last_decile01": float(ps[-n10:].mean()),
        "tail15": float(np.sort(ps)[-n15:].mean()),
        "regretful_rate": float(r.regretful.mean()),
        "G_T01": r.G_T * r.to01,
        "weak01": float(r.weak_regret.sum() * r.to01),
        "seconds": r.seconds,
    }
    row.update(checkpoint_totals(ps, checkpoints))
    if extra:
        row.update(extra)
    return row


def i_conv_over_seeds(
    curves_by_seed: dict[int, tuple[np.ndarray, np.ndarray]],
    window: int = I_CONV_WINDOW,
) -> dict:
    """curves_by_seed[seed] = (learner_per_student01, oracle_per_student01)."""
    vals = [i_conv(a, b, window=window) for a, b in curves_by_seed.values()]
    present = [v for v in vals if v is not None]
    return {
        "window": window,
        "per_seed": {str(s): v for s, v in zip(curves_by_seed, vals)},
        "median": float(np.median(present)) if present else None,
        "n_converged": len(present),
        "n_seeds": len(vals),
    }


def summary_bundle(rows: list[dict], competitor: str = "none:emax") -> dict:
    out = {"metrics": {}}
    for key in ("total01", "G_T01", "weak01", "tail15", "regretful_rate"):
        if rows and key in rows[0]:
            out["metrics"][key] = summarize_runs(rows, key)
    out["vs_competitor"] = competitor
    paired = out["metrics"].get("total01", {}).get("paired", {})
    # keys are "a - b" lexicographic
    hits = [v for k, v in paired.items() if competitor in k]
    out["paired_vs_competitor"] = hits
    return out
