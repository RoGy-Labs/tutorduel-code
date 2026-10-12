"""Summaries and paired comparisons over frozen seeds."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    se = float(x.std(ddof=1) / np.sqrt(n))
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * se
    return m, se, h


def summarize_runs(rows: list[dict], key: str = "total01") -> dict:
    """rows: list of dicts with 'policy', 'seed', and metric columns. Returns per-policy
    mean / se / ci half-width of `key`, plus paired differences vs each other policy."""
    pols = sorted({r["policy"] for r in rows})
    by = {p: {r["seed"]: r[key] for r in rows if r["policy"] == p} for p in pols}
    out = {"metric": key, "policies": {}, "paired": {}}
    for p in pols:
        vals = np.array(list(by[p].values()))
        m, se, h = mean_ci(vals)
        out["policies"][p] = {"mean": m, "se": se, "ci95": h, "n": len(vals)}
    for p in pols:
        for q in pols:
            if p >= q:
                continue
            seeds = sorted(set(by[p]) & set(by[q]))
            if len(seeds) < 2:
                continue
            diff = np.array([by[p][s] - by[q][s] for s in seeds])
            m, se, h = mean_ci(diff)
            out["paired"][f"{p} - {q}"] = {"mean": m, "se": se, "ci95": h, "n": len(seeds),
                                          "frac_better": float(np.mean(diff < 0))}
    return out


def learning_curves(regret_by_seed: list[np.ndarray], to01: float = 1.0) -> dict:
    """regret_by_seed: list of (N,b) arrays. Returns mean per-student regret curve (N,),
    mean within-student curve (b,), and cumulative curves."""
    R = np.stack(regret_by_seed) * to01                 # (n_seeds, N, b)
    per_student = R.sum(axis=2)                          # (n_seeds, N)
    within = R.mean(axis=1)                              # (n_seeds, b)
    return {
        "per_student_mean": per_student.mean(axis=0),
        "per_student_se": per_student.std(axis=0, ddof=1) / np.sqrt(R.shape[0]) if R.shape[0] > 1 else np.zeros(R.shape[1]),
        "within_mean": within.mean(axis=0),
        "cum_total_mean": per_student.cumsum(axis=1).mean(axis=0),
    }
