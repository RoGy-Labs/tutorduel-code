"""Uncertainty for paired comparisons: seed-only t interval, crossed seed x arrival bootstrap (stratified by profile
type), and minimum detectable effects. Generalises results/tstat.py."""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def t_interval(d, conf=0.95):
    d = np.asarray(d, float); n = len(d); m = float(d.mean())
    if n < 2:
        return dict(mean=m, se=float("nan"), lower=float("nan"), upper=float("nan"), n=n)
    se = float(d.std(ddof=1) / np.sqrt(n)); h = float(st.t.ppf(0.5 + conf / 2, n - 1) * se)
    return dict(mean=m, se=se, lower=m - h, upper=m + h, n=n, t=m / max(d.std(ddof=1), 1e-12), p=float(2 * st.t.sf(abs(m / max(se, 1e-12)), n - 1)))


def mde(sd, n, alpha=0.05, power=0.8):
    """smallest true mean difference detected with the given power by a two-sided paired t-test on n seeds."""
    return float((st.t.ppf(1 - alpha / 2, n - 1) + st.t.ppf(power, n - 1)) * sd / np.sqrt(n))


def crossed_bootstrap(D, strata=None, n_boot=2000, seed=0, conf=0.95):
    """D[s, a]: paired difference of seed s on arrival a (every seed visits every arrival of a replay roster).
    Resamples seeds AND arrivals (arrivals within strata), statistic = mean over seeds of the roster total.
    The seed-only interval treats the roster as fixed; this one also answers 'with another roster of the same kind'."""
    D = np.asarray(D, float); S, A = D.shape
    rng = np.random.default_rng(seed)
    strata = np.zeros(A, int) if strata is None else np.asarray(strata)
    groups = [np.where(strata == g)[0] for g in np.unique(strata)]
    out = np.empty(n_boot)
    for k in range(n_boot):
        ss = rng.integers(0, S, S)
        aa = np.concatenate([rng.choice(g, len(g)) for g in groups])
        out[k] = D[np.ix_(ss, aa)].sum(1).mean()
    lo, hi = np.quantile(out, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return dict(mean=float(D.sum(1).mean()), lower=float(lo), upper=float(hi), se=float(out.std(ddof=1)), n_seeds=S, n_arrivals=A)


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: boolean mask of discoveries."""
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    thr = q * (np.arange(1, n + 1) / n)
    ok = p[order] <= thr
    k = (np.where(ok)[0].max() + 1) if ok.any() else 0
    mask = np.zeros(n, bool); mask[order[:k]] = True
    return mask


def holm(pvals, alpha=0.05):
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    mask = np.zeros(n, bool)
    for r, k in enumerate(order):
        if p[k] <= alpha / (n - r):
            mask[k] = True
        else:
            break
    return mask
